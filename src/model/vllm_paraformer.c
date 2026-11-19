/* Independent FP32 ParaformerStreaming VQF v7 runner. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <setjmp.h>
#include <limits.h>
#ifndef VLLM_PARAFORMER_NO_MAIN
#include <errno.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif
#endif
#include "../../include/model/vllm_paraformer.h"
#include "vllm_paraformer_pv.h"
#include "vllm_paraformer_linear.h"
#define D 512
#define MAXT 128
#define VOC 8404
/* Allocation scope rolls back all scratch allocations on error. Persistent
 * allocations are explicitly detached before ownership passes to a handle. */
typedef struct Allocation { void *p; struct Allocation *next; } Allocation;
typedef struct { jmp_buf jump; Allocation *allocations; char error[160]; } Scope;
static _Thread_local Scope *active_scope;
static void die(const char *s) { snprintf(active_scope->error,sizeof(active_scope->error),"%s",s);longjmp(active_scope->jump,1); }
static void *mem(size_t n) { void *p=calloc(1,n?n:1);Allocation *a=malloc(sizeof(*a));if(!p||!a){free(p);free(a);die("allocation failed");}a->p=p;a->next=active_scope->allocations;active_scope->allocations=a;return p; }
static void detach(void *p) { Allocation **a=&active_scope->allocations;while(*a){if((*a)->p==p){Allocation *old=*a;*a=old->next;free(old);return;}a=&(*a)->next;} }
static void release(void *p) { if(p){detach(p);free(p);} }
static void scope_clear(Scope *s) { while(s->allocations){Allocation *a=s->allocations;s->allocations=a->next;free(a->p);free(a);} }
static uint32_t u32(const unsigned char *p) { return p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24; }
static uint64_t u64(const unsigned char *p) { return u32(p)|(uint64_t)u32(p+4)<<32; }
#ifndef VLLM_PARAFORMER_NO_MAIN
static void rd(FILE *f,void *p,size_t n) { if(fread(p,1,n,f)!=n)die("truncated input"); }
#endif
typedef struct { char name[32];uint32_t type,r,c;uint64_t off,len; } Tensor;
struct vllm_paraformer_model { unsigned char *data;size_t size;Tensor ts[960];int nt;char *manifest;size_t refs;float melbank[80][257],window400[400]; };
typedef struct vllm_paraformer_model Model;
static Tensor *entry(Model *m,const char *name) { for(int i=0;i<m->nt;i++)if(!strcmp(m->ts[i].name,name))return m->ts+i;die("tensor not found");return NULL; }
static const float *weight(Model *m,const char *prefix,const char *suffix,int rows,int cols) {
    char key[256],id[32];snprintf(key,sizeof(key),"\"%s%s\":{\"name\":\"",prefix,suffix);
    char *p=strstr(m->manifest,key);if(!p)die("manifest missing tensor");p+=strlen(key);int n=0;while(*p&&*p!='\"'&&n<31)id[n++]=*p++;id[n]=0;if(*p!='\"')die("invalid tensor ID");
    Tensor *t=entry(m,id);if(t->type!=1||t->r!=(unsigned)rows||t->c!=(unsigned)cols||t->len!=(uint64_t)rows*cols*4)die("tensor shape/type mismatch");return (const float *)(m->data+t->off);
}
static void check_norm(Model *m,const char *p,int d) { weight(m,p,".weight",d,1);weight(m,p,".bias",d,1); }
static void check_linear(Model *m,const char *p,int out,int in,int bias) { weight(m,p,".weight",out,in);if(bias)weight(m,p,".bias",out,1); }
static void validate_weights(Model *m) {
    char p[128],q[180];
    for(int l=0;l<50;l++){
        if(l)snprintf(p,sizeof(p),"encoder.encoders.%d",l-1);else strcpy(p,"encoder.encoders0.0");
        snprintf(q,sizeof(q),"%s.norm1",p);check_norm(m,q,l?512:560);snprintf(q,sizeof(q),"%s.norm2",p);check_norm(m,q,512);
        snprintf(q,sizeof(q),"%s.self_attn.linear_q_k_v",p);check_linear(m,q,1536,l?512:560,1);snprintf(q,sizeof(q),"%s.self_attn.linear_out",p);check_linear(m,q,512,512,1);weight(m,p,".self_attn.fsmn_block.weight",512,11);
        snprintf(q,sizeof(q),"%s.feed_forward.w_1",p);check_linear(m,q,2048,512,1);snprintf(q,sizeof(q),"%s.feed_forward.w_2",p);check_linear(m,q,512,2048,1);
    }
    for(int l=0;l<17;l++){
        if(l<16)snprintf(p,sizeof(p),"decoder.decoders.%d",l);else strcpy(p,"decoder.decoders3.0");
        snprintf(q,sizeof(q),"%s.norm1",p);check_norm(m,q,512);snprintf(q,sizeof(q),"%s.feed_forward.norm",p);check_norm(m,q,2048);snprintf(q,sizeof(q),"%s.feed_forward.w_1",p);check_linear(m,q,2048,512,1);snprintf(q,sizeof(q),"%s.feed_forward.w_2",p);check_linear(m,q,512,2048,0);
        if(l<16){snprintf(q,sizeof(q),"%s.norm2",p);check_norm(m,q,512);snprintf(q,sizeof(q),"%s.norm3",p);check_norm(m,q,512);weight(m,p,".self_attn.fsmn_block.weight",512,11);snprintf(q,sizeof(q),"%s.src_attn.linear_q",p);check_linear(m,q,512,512,1);snprintf(q,sizeof(q),"%s.src_attn.linear_k_v",p);check_linear(m,q,1024,512,1);snprintf(q,sizeof(q),"%s.src_attn.linear_out",p);check_linear(m,q,512,512,1);}
    }
    check_norm(m,"encoder.after_norm",512);check_norm(m,"decoder.after_norm",512);check_linear(m,"decoder.output_layer",8404,512,1);weight(m,"decoder.embed.0",".weight",8404,512);check_linear(m,"predictor.cif_conv1d",512,1536,1);check_linear(m,"predictor.cif_output",1,512,1);weight(m,"frontend.cmvn_shift","",560,1);weight(m,"frontend.cmvn_scale","",560,1);
}
static void load(Model *m,const char *path) {
    FILE *f=fopen(path,"rb");if(!f)die("cannot open VQF");
    if(fseek(f,0,SEEK_END)){fclose(f);die("seek failed");}long sz=ftell(f);if(sz<448||sz>1500000000L){fclose(f);die("invalid VQF size");}rewind(f);
    /* Allocate without throwing while the FILE is open. */
    m->data=calloc(1,(size_t)sz);if(!m->data){fclose(f);die("allocation failed");}m->size=(size_t)sz;size_t got=fread(m->data,1,m->size,f);int closed=fclose(f);if(got!=m->size||closed)die("truncated VQF");
    unsigned char *h=m->data;uint32_t nt=u32(h+12);uint64_t start=u64(h+200),end=u64(h+208),directory_end=448+64*(uint64_t)nt;
    if(u32(h)!=0x57465146||u32(h+4)!=7||u32(h+8)!=(1u<<13)||memcmp(h+16,"PARAFORMER-v1",12)||nt!=960||directory_end>m->size-8||start<directory_end||end!=m->size-8||start>end){die("invalid Paraformer v7 header");}
    m->nt=(int)nt;
    uint64_t prev=start;
    for(int i=0;i<m->nt;i++){unsigned char *p=h+448+64*i;Tensor *t=m->ts+i;memcpy(t->name,p,32);if(!memchr(t->name,0,32)||!t->name[0])die("invalid tensor name");t->type=u32(p+32);t->r=u32(p+36);t->c=u32(p+40);t->off=u64(p+48);t->len=u64(p+56);if(t->off<prev||t->off>end||t->len>end-t->off||t->off%64||!t->len)die("invalid tensor range");prev=t->off+t->len;for(int j=0;j<i;j++)if(!strcmp(t->name,m->ts[j].name))die("duplicate tensor name");if(t->type==1){if(!t->r||!t->c||(uint64_t)t->r*t->c>UINT64_MAX/4||t->len!=(uint64_t)t->r*t->c*4)die("invalid FP32 tensor");}else if(t->type!=6)die("unsupported tensor type");}
    uint64_t hash=UINT64_C(14695981039346656037);for(size_t i=448;i<directory_end;i++)hash=(hash^h[i])*UINT64_C(1099511628211);for(uint64_t i=start;i<end;i++)hash=(hash^h[i])*UINT64_C(1099511628211);if(hash!=u64(h+end))die("VQF checksum mismatch");
    for(int i=0;i<m->nt;i++)if(m->ts[i].type==1){const float *v=(const float *)(h+m->ts[i].off);for(size_t j=0;j<m->ts[i].len/4;j++)if(!isfinite(v[j]))die("nonfinite weight");}
    Tensor *t=entry(m,"pf_manifest");if(t->type!=6||t->len>1000000||memchr(h+t->off,0,(size_t)t->len))die("invalid manifest");m->manifest=calloc(1,(size_t)t->len+1);if(!m->manifest)die("allocation failed");memcpy(m->manifest,h+t->off,(size_t)t->len);if(!strstr(m->manifest,"\"family\":\"paraformer-streaming\",\"schema\":1"))die("unsupported manifest");
    t=entry(m,"pf_tokens");if(t->type!=6||t->len>1000000)die("invalid tokens");
    const unsigned char *p=h+t->off,*stop=p+t->len;if(*p++!='[')die("invalid tokens");int tokens=0;
    while(p<stop&&*p!=']'){if(tokens&&*p++!=',')die("invalid tokens");if(p==stop||*p++!='\"')die("invalid tokens");while(p<stop&&*p!='\"'){if(*p<32)die("invalid token string");if(*p++=='\\'){if(p==stop)die("invalid token escape");unsigned char c=*p++;if(c=='u'){for(int k=0;k<4;k++){if(p==stop||!((*p>='0'&&*p<='9')||(*p>='a'&&*p<='f')||(*p>='A'&&*p<='F')))die("invalid token escape");p++;}}else if(!strchr("\"\\/bfnrt",c))die("invalid token escape");}}if(p==stop)die("invalid token string");p++;tokens++;}
    if(tokens!=VOC||p==stop||*p++!=']'||p!=stop){die("invalid vocabulary");}
    validate_weights(m);
}
static void linear(Model *m,float *y,const float *x,int n,int in,int out,const char *p,const char *s,int bias) {
    char name[256];snprintf(name,sizeof(name),"%s%s",p,s);const float *w=weight(m,name,".weight",out,in),*b=bias?weight(m,name,".bias",out,1):NULL;
    pf_linear(y,x,w,b,n,in,out);
}
static void norm(Model *m,float *y,const float *x,int n,int d,const char *p,const char *s) {
    char name[256];snprintf(name,sizeof(name),"%s%s",p,s);const float *w=weight(m,name,".weight",d,1),*b=weight(m,name,".bias",d,1);
    for(int t=0;t<n;t++){double sum=0,var=0;for(int j=0;j<d;j++)sum+=x[t*d+j];float mean=(float)(sum/d);for(int j=0;j<d;j++){double a=x[t*d+j]-mean;var+=a*a;}float inv=1.0f/sqrtf((float)(var/d)+1e-12f);for(int j=0;j<d;j++)y[t*d+j]=(x[t*d+j]-mean)*inv*w[j]+b[j];}
}
static void ffn(Model *m,float *y,const float *x,int n,const char *p,int dec) {
    float *z=mem((size_t)n*2048*4);linear(m,z,x,n,D,2048,p,".feed_forward.w_1",1);for(int i=0;i<n*2048;i++)z[i]=fmaxf(z[i],0);if(dec)norm(m,z,z,n,2048,p,".feed_forward.norm");linear(m,y,z,n,2048,D,p,".feed_forward.w_2",!dec);release(z);
}
typedef struct { float *k,*v;int n; } KV;
typedef struct { KV enc[50],dec[16];float fsmn[16][10*D];float overlap[5*560];float cif[D],alpha;int position; } State;
static void reset(State *s) { for(int i=0;i<50;i++){free(s->enc[i].k);free(s->enc[i].v);}for(int i=0;i<16;i++){free(s->dec[i].k);free(s->dec[i].v);}memset(s,0,sizeof(*s)); }
static void attention(float *out,const float *q,const float *k,const float *v,int nq,int nk,KV *cache,int keep,int append) {
    int old=cache->n,total=old+nk;float *kk=mem((size_t)total*D*4),*vv=mem((size_t)total*D*4),*scores=mem((size_t)nq*total*4);
    if(old){memcpy(kk,cache->k,(size_t)old*D*4);memcpy(vv,cache->v,(size_t)old*D*4);}memcpy(kk+old*D,k,(size_t)nk*D*4);memcpy(vv+old*D,v,(size_t)nk*D*4);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int t=0;t<nq;t++){float *score=scores+t*total;for(int h=0;h<4;h++){float mx=-FLT_MAX,sum=0;for(int j=0;j<total;j++){float a=0;for(int c=0;c<128;c++)a+=(q[t*D+h*128+c]*0.08838834764831845f)*kk[j*D+h*128+c];score[j]=a;if(a>mx)mx=a;}for(int j=0;j<total;j++){score[j]=expf(score[j]-mx);sum+=score[j];}pf_pv(out+t*D+h*128,score,sum,vv+h*128,total,128,D);}}
    if(keep){int available=old+append,sz=available<keep?available:keep;float *ck=mem((size_t)sz*D*4),*cv=mem((size_t)sz*D*4);memcpy(ck,kk+(available-sz)*D,(size_t)sz*D*4);memcpy(cv,vv+(available-sz)*D,(size_t)sz*D*4);free(cache->k);free(cache->v);detach(ck);detach(cv);cache->k=ck;cache->v=cv;cache->n=sz;}release(scores);release(kk);release(vv);
}
static void encoder(Model *m,State *s,const float *features,int nf,int tail,float *out,int *nt,int elook) {
    int n=tail?5:nf+5;if(n>MAXT||nf<0||s->position>INT_MAX-nf)die("invalid feature chunk");float *x=mem((size_t)n*560*4),*z=mem((size_t)n*560*4),*a=mem((size_t)n*D*4),*b=mem((size_t)n*D*4);
    memcpy(x,s->overlap,sizeof(s->overlap));if(!tail){for(int t=0;t<nf;t++)for(int c=0;c<560;c++){float time=(float)(s->position+t+1)*expf((float)(c%280)*(-logf(10000.0f)/279.0f));x[(t+5)*560+c]=features[t*560+c]*sqrtf(512.0f)+(c<280?sinf(time):cosf(time));}s->position+=nf;memcpy(s->overlap,x+(n-5)*560,sizeof(s->overlap));}
    for(int l=0;l<50;l++){char p[128];if(!l)strcpy(p,"encoder.encoders0.0");else snprintf(p,sizeof(p),"encoder.encoders.%d",l-1);int width=l?D:560;norm(m,z,x,n,width,p,".norm1");float *qkv=mem((size_t)n*1536*4),*q=mem((size_t)n*D*4),*k=mem((size_t)n*D*4),*v=mem((size_t)n*D*4);linear(m,qkv,z,n,width,1536,p,".self_attn.linear_q_k_v",1);for(int t=0;t<n;t++){memcpy(q+t*D,qkv+t*1536,D*4);memcpy(k+t*D,qkv+t*1536+D,D*4);memcpy(v+t*D,qkv+t*1536+2*D,D*4);}attention(a,q,k,v,n,n,&s->enc[l],elook*10,n-5);linear(m,b,a,n,D,D,p,".self_attn.linear_out",1);const float *w=weight(m,p,".self_attn.fsmn_block.weight",D,11);
        for(int t=0;t<n;t++)for(int c=0;c<D;c++){float f=0;for(int j=0;j<11;j++){int at=t+j-5;if(at>=0&&at<n)f+=v[at*D+c]*w[c*11+j];}b[t*D+c]+=f+v[t*D+c];if(l)b[t*D+c]+=x[t*D+c];}
        norm(m,z,b,n,D,p,".norm2");ffn(m,a,z,n,p,0);for(int i=0;i<n*D;i++)x[i]=b[i]+a[i];release(qkv);release(q);release(k);release(v);
    }norm(m,out,x,n,D,"encoder",".after_norm");*nt=n;release(x);release(z);release(a);release(b);
}
static int predictor(Model *m,State *s,const float *x,int n,int final,float *emb,float *alphas) {
    const float *w=weight(m,"predictor.cif_conv1d",".weight",D,D*3),*b=weight(m,"predictor.cif_conv1d",".bias",D,1),*ow=weight(m,"predictor.cif_output",".weight",1,D),*ob=weight(m,"predictor.cif_output",".bias",1,1);
    for(int t=0;t<n;t++){float sum=0;for(int o=0;o<D;o++){float a=0;for(int c=0;c<D;c++)for(int j=0;j<3;j++){int at=t+j-1;if(at>=0&&at<n)a+=x[at*D+c]*w[(o*D+c)*3+j];}sum+=fmaxf(a+b[o],0)*ow[o];}alphas[t]=(!final&&t>=10)?0:1.0f/(1.0f+expf(-(sum+ob[0])));}
    float frames[D];for(int c=0;c<D;c++)frames[c]=s->cif[c]*s->alpha;float integrate=s->alpha;int count=0;
    for(int t=0;t<n+final;t++){float alpha=t<n?alphas[t]:0.45f;const float *h=t<n?x+t*D:NULL;if(alpha+integrate<1){integrate+=alpha;for(int c=0;c<D;c++)frames[c]+=alpha*(h?h[c]:0);}else{for(int c=0;c<D;c++)emb[count*D+c]=frames[c]+(1-integrate)*(h?h[c]:0);count++;integrate+=alpha;integrate-=1;for(int c=0;c<D;c++)frames[c]=integrate*(h?h[c]:0);}}
    s->alpha=integrate;for(int c=0;c<D;c++)s->cif[c]=integrate>0?frames[c]/integrate:frames[c];return count;
}
static void decoder(Model *m,State *s,const float *enc,int ne,float *x,int n,float *logits,int dlook) {
    float *z=mem((size_t)n*D*4),*a=mem((size_t)n*D*4),*b=mem((size_t)n*D*4);
    for(int l=0;l<17;l++){char p[128];if(l<16)snprintf(p,sizeof(p),"decoder.decoders.%d",l);else strcpy(p,"decoder.decoders3.0");norm(m,z,x,n,D,p,".norm1");ffn(m,a,z,n,p,1);if(l==16){memcpy(x,a,(size_t)n*D*4);break;}norm(m,z,a,n,D,p,".norm2");const float *w=weight(m,p,".self_attn.fsmn_block.weight",D,11);float *history=mem((size_t)(n+10)*D*4);memcpy(history,s->fsmn[l],10*D*4);memcpy(history+10*D,z,(size_t)n*D*4);
        for(int t=0;t<n;t++)for(int c=0;c<D;c++){float sum=0;for(int j=0;j<11;j++)sum+=history[(t+j)*D+c]*w[c*11+j];x[t*D+c]+=sum+z[t*D+c];}
        memcpy(s->fsmn[l],history+n*D,10*D*4);release(history);
        norm(m,z,x,n,D,p,".norm3");linear(m,a,z,n,D,D,p,".src_attn.linear_q",1);float *kv=mem((size_t)ne*1024*4),*k=mem((size_t)ne*D*4),*v=mem((size_t)ne*D*4);linear(m,kv,enc,ne,D,1024,p,".src_attn.linear_k_v",1);for(int t=0;t<ne;t++){memcpy(k+t*D,kv+t*1024,D*4);memcpy(v+t*D,kv+t*1024+D,D*4);}attention(b,a,k,v,n,ne,&s->dec[l],dlook*10,ne);linear(m,a,b,n,D,D,p,".src_attn.linear_out",1);for(int i=0;i<n*D;i++)x[i]+=a[i];release(kv);release(k);release(v);
    }norm(m,z,x,n,D,"decoder",".after_norm");linear(m,logits,z,n,D,VOC,"decoder",".output_layer",1);release(z);release(a);release(b);
}
#ifndef VLLM_PARAFORMER_NO_MAIN
static void wr(FILE *f,const void *p,size_t n) { if(fwrite(p,1,n,f)!=n)die("output write failed"); }
#endif
#include "vllm_paraformer_frontend.h"

struct vllm_paraformer_session { Model *model;State state;Frontend front;float pending[9600];int np,done,busy;vllm_paraformer_output output;void *user;char error[160]; };
int vllm_paraformer_model_create(const char *path,Model **out,char *err,size_t capacity) {
    if(out){*out=NULL;}
    if(!path||!out||(!err&&capacity))return VLLM_PF_INVALID;
    if(capacity)err[0]=0;
    Model *m=calloc(1,sizeof(*m));Scope *scope=calloc(1,sizeof(*scope));if(!m||!scope){free(m);free(scope);return VLLM_PF_ERROR;}Scope *previous=active_scope;active_scope=scope;int result;
    if(setjmp(scope->jump)){if(capacity)snprintf(err,capacity,"%s",scope->error);free(m->manifest);free(m->data);free(m);result=VLLM_PF_ERROR;}else{load(m,path);frontend_init(m);m->refs=1;*out=m;result=VLLM_PF_OK;}scope_clear(scope);active_scope=previous;free(scope);return result;
}
void vllm_paraformer_model_destroy(Model *m) { if(m&&!--m->refs){free(m->manifest);free(m->data);free(m);} }
int vllm_paraformer_session_create(Model *m,vllm_paraformer_output output,void *user,vllm_paraformer_session **out) { if(out)*out=NULL;if(!m||!out)return VLLM_PF_INVALID;vllm_paraformer_session *s=calloc(1,sizeof(*s));if(!s)return VLLM_PF_ERROR;s->model=m;m->refs++;s->output=output;s->user=user;*out=s;return 0; }
static void session_chunk(vllm_paraformer_session *s,int final) {
    float features[MAXT*560];int tail=final&&s->np<960;int nf=tail?0:frontend(s->model,&s->front,s->pending,s->np,final,features);
    float *enc=mem(MAXT*D*4),*emb=mem((MAXT+2)*D*4),alpha[MAXT];int ne;encoder(s->model,&s->state,features,nf,tail,enc,&ne,4);int np=predictor(s->model,&s->state,enc,ne,final,emb,alpha);float *logits=mem((size_t)np*VOC*4);int ids[MAXT+2];if(np)decoder(s->model,&s->state,enc,ne,emb,np,logits,1);for(int t=0;t<np;t++){int best=0;for(int i=1;i<VOC;i++)if(logits[t*VOC+i]>logits[t*VOC+best])best=i;ids[t]=best;}if(s->output)s->output(s->user,ids,(size_t)np,np?logits:NULL);release(enc);release(emb);release(logits);s->np=0;
}
static void session_process(vllm_paraformer_session *s,const int16_t *pcm,size_t samples,int final) {
    while(samples){size_t take=9600-s->np;if(take>samples)take=samples;for(size_t i=0;i<take;i++)s->pending[s->np+(int)i]=(float)pcm[i];s->np+=(int)take;pcm+=take;samples-=take;if(s->np==9600)session_chunk(s,0);}if(final){session_chunk(s,1);s->done=1;}
}
static int session_run(vllm_paraformer_session *s,const int16_t *pcm,size_t samples,int final) {
    if(!s||(!pcm&&samples))return VLLM_PF_INVALID;
    if(s->done||s->busy)return VLLM_PF_STATE;
    Scope *scope=calloc(1,sizeof(*scope));if(!scope)return VLLM_PF_ERROR;Scope *previous=active_scope;active_scope=scope;s->busy=1;int result;
    if(setjmp(scope->jump)){snprintf(s->error,sizeof(s->error),"%s",scope->error);s->done=1;result=VLLM_PF_ERROR;}else{session_process(s,pcm,samples,final);result=VLLM_PF_OK;}
    scope_clear(scope);active_scope=previous;free(scope);s->busy=0;return result;
}
int vllm_paraformer_session_feed(vllm_paraformer_session *s,const int16_t *pcm,size_t n) { return session_run(s,pcm,n,0); }
int vllm_paraformer_session_finish(vllm_paraformer_session *s) { return session_run(s,NULL,0,1); }
int vllm_paraformer_session_reset(vllm_paraformer_session *s) { if(!s)return VLLM_PF_INVALID;if(s->busy)return VLLM_PF_STATE;reset(&s->state);memset(&s->front,0,sizeof(s->front));memset(s->pending,0,sizeof(s->pending));s->np=s->done=0;s->error[0]=0;return 0; }
const char *vllm_paraformer_session_error(const vllm_paraformer_session *s) { return s?s->error:"invalid session"; }
void vllm_paraformer_session_destroy(vllm_paraformer_session *s) { if(s&&!s->busy){reset(&s->state);vllm_paraformer_model_destroy(s->model);free(s);} }
#ifndef VLLM_PARAFORMER_NO_MAIN
static void pcm_output(void *user,const int *ids,size_t count,const float *logits) {
    (void)logits;for(size_t i=0;i<count;i++)print_token((Model *)user,ids[i]);
    if(fflush(stdout)||ferror(stdout))die("output write failed");
}
static int pcm_run(Model *m) {
    vllm_paraformer_session *session=NULL;
    if(vllm_paraformer_session_create(m,pcm_output,m,&session))die("session allocation failed");
    unsigned char bytes[3201];int16_t pcm[1600];size_t pending=0;int failed=0;const char *error=NULL;
#ifdef _WIN32
    if(_setmode(0,_O_BINARY)==-1){error="cannot set binary stdin";failed=1;}
#endif
    while(!failed){
#ifdef _WIN32
        int got=_read(0,bytes+pending,3200);
#else
        ssize_t got=read(STDIN_FILENO,bytes+pending,3200);
#endif
        if(got<0){if(errno==EINTR)continue;error="stdin read failed";failed=1;break;}
        if(!got){if(pending){error="incomplete PCM16 sample at EOF";failed=1;}break;}
        size_t n=pending+(size_t)got,samples=n/2;
        for(size_t i=0;i<samples;i++){int v=bytes[2*i]|bytes[2*i+1]<<8;pcm[i]=(int16_t)(v>=32768?v-65536:v);}
        pending=n%2;if(pending)bytes[0]=bytes[n-1];
        if(vllm_paraformer_session_feed(session,pcm,samples))failed=1;
    }
    if(!failed&&vllm_paraformer_session_finish(session))failed=1;
    if(failed)fprintf(stderr,"paraformer: %s\n",error?error:vllm_paraformer_session_error(session));
    vllm_paraformer_session_destroy(session);
    if(!failed&&(putchar('\n')==EOF||fflush(stdout)))failed=1;
    return failed;
}
static int cli(Model *m,int argc,char **argv,FILE *f,FILE *o,State *s) {
    if(argc==3)return !strcmp(argv[2],"--pcm-stdin")?pcm_run(m):wav_run(m,argv[2]);
    char magic[8];rd(f,magic,8);if(memcmp(magic,"PFCH0001",8))die("invalid feature stream");wr(o,"PFOUT001",8);unsigned char h[16];size_t header_bytes;
    while((header_bytes=fread(h,1,16,f))==16){int nf=(int)u32(h),final=(int)u32(h+4),tail=(int)u32(h+8);if(u32(h)>MAXT-5||u32(h+4)>1||u32(h+8)>1||u32(h+12)>1)die("invalid chunk");if(u32(h+12))reset(s);float *feat=mem((size_t)nf*560*4);rd(f,feat,(size_t)nf*560*4);for(int i=0;i<nf*560;i++)if(!isfinite(feat[i]))die("nonfinite feature");float *enc=mem(MAXT*D*4),*emb=mem((MAXT+2)*D*4),alpha[MAXT];int ne;encoder(m,s,feat,nf,tail,enc,&ne,4);int np=predictor(m,s,enc,ne,final,emb,alpha);float *logits=mem((size_t)np*VOC*4);uint32_t dims[2]={(uint32_t)ne,(uint32_t)np};wr(o,dims,8);wr(o,enc,(size_t)ne*D*4);wr(o,alpha,(size_t)ne*4);wr(o,emb,(size_t)np*D*4);if(np)decoder(m,s,enc,ne,emb,np,logits,1);wr(o,logits,(size_t)np*VOC*4);fprintf(stderr,"chunk frames=%d tokens=%d final=%d\n",ne,np,final);release(feat);release(enc);release(emb);release(logits);}if(header_bytes||ferror(f))die("truncated or unreadable feature stream");return 0;
}
int main(int argc,char **argv) {
    if(argc!=3&&argc!=4){fprintf(stderr,"usage: paraformer model.vqf audio.wav | model.vqf --pcm-stdin | model.vqf features.pfch output.pfout\n");return 2;}Model *m=NULL;char error[160];if(vllm_paraformer_model_create(argv[1],&m,error,sizeof(error))){fprintf(stderr,"paraformer: %s\n",error);return 1;}
    FILE *f=argc==4?fopen(argv[2],"rb"):NULL,*o=argc==4?fopen(argv[3],"wb"):NULL;Scope *scope=calloc(1,sizeof(*scope));State *s=calloc(1,sizeof(*s));if(!scope||!s||(argc==4&&(!f||!o))){if(f)fclose(f);if(o)fclose(o);free(scope);free(s);vllm_paraformer_model_destroy(m);return 1;}active_scope=scope;int result;if(setjmp(scope->jump)){fprintf(stderr,"paraformer: %s\n",scope->error);result=1;}else result=cli(m,argc,argv,f,o,s);reset(s);free(s);scope_clear(scope);free(scope);active_scope=NULL;if(f)fclose(f);if(o&&fclose(o))result=1;vllm_paraformer_model_destroy(m);return result;
}
#endif

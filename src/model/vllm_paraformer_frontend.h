/* Kaldi-compatible deterministic (dither=0) 16 kHz PCM16 frontend. */
typedef struct { float pcm[10000],banks[80*80];int np,nb,started; } Frontend;
static void frontend_init(Model *m) {
    for(int i=0;i<400;i++)m->window400[i]=0.54f-0.46f*cosf(2.0f*3.14159265358979323846f*i/399);
    float low=1127.0f*logf(1+20.0f/700),high=1127.0f*logf(1+8000.0f/700),step=(high-low)/81;
    for(int j=0;j<80;j++)for(int k=0;k<256;k++){float mel=1127.0f*logf(1+(16000.0f/512*k)/700);float a=(mel-low-j*step)/step,b=(low+(j+2)*step-mel)/step;m->melbank[j][k]=fmaxf(0,fminf(a,b));}
}
static void fbank(Model *m,const float *pcm,float *out) {
    float re[512]={0},im[512]={0},mean=0;for(int i=0;i<400;i++)mean+=pcm[i];mean/=400;
    for(int i=0;i<400;i++){float x=pcm[i]-mean,prev=pcm[i?i-1:0]-mean;re[i]=(x-0.97f*prev)*m->window400[i];}
    for(int i=1,j=0;i<512;i++){int bit=256;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){float v=re[i];re[i]=re[j];re[j]=v;}}
    for(int len=2;len<=512;len*=2)for(int base=0;base<512;base+=len)for(int j=0;j<len/2;j++){float angle=-2.0f*3.14159265358979323846f*j/len,c=cosf(angle),s=sinf(angle);int a=base+j,b=a+len/2;float r=re[b]*c-im[b]*s,v=re[b]*s+im[b]*c;re[b]=re[a]-r;im[b]=im[a]-v;re[a]+=r;im[a]+=v;}
    for(int j=0;j<80;j++){float sum=0;for(int k=0;k<257;k++)sum+=m->melbank[j][k]*(re[k]*re[k]+im[k]*im[k]);out[j]=logf(fmaxf(sum,FLT_EPSILON));}
}
static int frontend(Model *m,Frontend *s,const float *pcm,int n,int final,float *out) {
    if(n+s->np>10000){die("PCM chunk too large");}memcpy(s->pcm+s->np,pcm,(size_t)n*4);s->np+=n;int frames=s->np>=400?(s->np-400)/160+1:0;
    for(int i=0;i<frames;i++){float bank[80];fbank(m,s->pcm+i*160,bank);if(!s->started){for(int j=0;j<3;j++)memcpy(s->banks+(s->nb++)*80,bank,80*4);s->started=1;}if(s->nb>=80)die("frontend frame overflow");memcpy(s->banks+(s->nb++)*80,bank,80*4);}
    memmove(s->pcm,s->pcm+frames*160,(size_t)(s->np-frames*160)*4);s->np-=frames*160;int count=final?(s->nb>3?(s->nb-3+5)/6:0):(s->nb>=7?(s->nb-7)/6+1:0);
    const float *shift=weight(m,"frontend.cmvn_shift","",560,1),*scale=weight(m,"frontend.cmvn_scale","",560,1);
    for(int t=0;t<count;t++)for(int j=0;j<7;j++){int at=t*6+j;if(at>=s->nb)at=s->nb-1;for(int c=0;c<80;c++){int d=j*80+c;out[t*560+d]=(s->banks[at*80+c]+shift[d])*scale[d];}}
    int used=count*6;if(used>=s->nb)used=s->nb?s->nb-1:0;memmove(s->banks,s->banks+used*80,(size_t)(s->nb-used)*80*4);s->nb-=used;return count;
}
#ifndef VLLM_PARAFORMER_NO_MAIN
static float *read_wav(FILE *f,int *n) {
    unsigned char h[16];rd(f,h,12);if(memcmp(h,"RIFF",4)||memcmp(h+8,"WAVE",4))die("expected RIFF WAVE");int fmt=0;float *pcm=NULL;size_t got;
    while((got=fread(h,1,8,f))==8){uint32_t size=u32(h+4);if(size>320000000)die("WAV chunk too large");if(!memcmp(h,"fmt ",4)){if(size<16)die("invalid WAV format");rd(f,h,16);if(h[0]!=1||h[1]||h[2]!=1||h[3]||u32(h+4)!=16000||h[14]!=16||h[15])die("expected mono 16kHz PCM16 WAV");fmt=1;if(fseek(f,size-16+(size&1),SEEK_CUR))die("WAV seek failed");}else if(!memcmp(h,"data",4)){if(!fmt||pcm||size%2)die("invalid WAV data");*n=(int)(size/2);pcm=mem((size_t)*n*4);for(int i=0;i<*n;i++){rd(f,h,2);int v=h[0]|h[1]<<8;pcm[i]=(float)(v>=32768?v-65536:v);}}else if(fseek(f,size+(size&1),SEEK_CUR))die("WAV seek failed");}if(got||ferror(f))die("truncated WAV");if(!pcm)die("missing WAV data");return pcm;
}
static void print_token(Model *m,int id) {
    if(id==0||id==1||id==2)return;
    Tensor *t=entry(m,"pf_tokens");const unsigned char *p=m->data+t->off,*end=p+t->len;int at=-1;
    while(p<end){if(*p++!='\"')continue;at++;const unsigned char *start=p;while(p<end&&*p!='\"'){if(*p=='\\'&&p+1<end)p++;p++;}if(p==end)die("invalid token JSON");if(at==id){for(const unsigned char *q=start;q<p;q++){if(*q=='\\'&&q+1<p){q++;if(*q=='n')putchar('\n');else if(*q=='t')putchar('\t');else putchar(*q);}else putchar(*q);}return;}p++;}die("token ID out of range");
}
static void wav_process(Model *m,FILE *f,FILE *dump,State *s) {
    int ns;float *pcm=read_wav(f,&ns);Frontend *front=mem(sizeof(Frontend));if(dump)wr(dump,"PFCH0001",8);
    for(int at=0;at<=ns;at+=9600){int count=ns-at;if(count>9600)count=9600;int final=at+9600>ns,tail=final&&count<960;float features[MAXT*560];int nf=tail?0:frontend(m,front,pcm+at,count,final,features);if(dump){uint32_t h[4]={(uint32_t)nf,(uint32_t)final,(uint32_t)tail,at==0};wr(dump,h,16);wr(dump,features,(size_t)nf*560*4);}float *enc=mem(MAXT*D*4),*emb=mem((MAXT+2)*D*4),alpha[MAXT];int ne;encoder(m,s,features,nf,tail,enc,&ne,4);int np=predictor(m,s,enc,ne,final,emb,alpha);float *logits=mem((size_t)np*VOC*4);if(np)decoder(m,s,enc,ne,emb,np,logits,1);for(int t=0;t<np;t++){int best=0;for(int i=1;i<VOC;i++)if(logits[t*VOC+i]>logits[t*VOC+best])best=i;print_token(m,best);}release(enc);release(emb);release(logits);if(final)break;}
}
static int wav_run(Model *m,const char *path) {
    /* Nested scope ensures FILEs and detached KV buffers close on every error. */
    FILE *f=fopen(path,"rb");if(!f)die("cannot open WAV");FILE *dump=NULL;const char *dump_path=getenv("KESTREL_PF_FEATURE_DUMP");if(dump_path){dump=fopen(dump_path,"wb");if(!dump){fclose(f);die("cannot open feature dump");}}
    Scope *scope=calloc(1,sizeof(*scope));State *s=calloc(1,sizeof(*s));if(!scope||!s){fclose(f);if(dump)fclose(dump);free(scope);free(s);die("allocation failed");}Scope *previous=active_scope;active_scope=scope;int failed;
    if(setjmp(scope->jump))failed=1;else{wav_process(m,f,dump,s);failed=0;}
    fclose(f);if(dump&&fclose(dump)){failed=1;snprintf(scope->error,sizeof(scope->error),"feature dump close failed");}reset(s);free(s);scope_clear(scope);char error[160];snprintf(error,sizeof(error),"%s",scope->error);free(scope);active_scope=previous;if(failed)die(error);putchar('\n');return 0;
}
#endif

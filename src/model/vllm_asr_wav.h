/* 独立 WAV 特征与 ByteLevel 解码；仅依赖 C11 标准库。 */
static unsigned char *wav_asset;
static float wav_window[400], wav_filters[201*128];
static uint32_t wav_prefix[CAP],wav_suffix[CAP],wav_np,wav_ns;
static unsigned char *wav_offsets,*wav_bytes;
static size_t wav_bytes_size;
static void wav_assets(const char *path,uint64_t model_digest) {
    int previous=prof_set(P_ASSETS);
    FILE *f=open_file(path,"rb");unsigned char h[32];read_exact(f,h,32);
    uint64_t size=u64(h+16);
    if(memcmp(h,"ASRWAV01",8) || (model_digest && u64(h+8)!=model_digest) || size<620000 || size>16000000) fail("invalid WAV assets/model binding");
    wav_asset=alloc((size_t)size);read_exact(f,wav_asset,(size_t)size);
    if(fgetc(f)!=EOF || hash(UINT64_C(14695981039346656037),wav_asset,(size_t)size)!=u64(h+24)) fail("WAV asset checksum/length mismatch");
    fclose(f);wav_np=u32(wav_asset);wav_ns=u32(wav_asset+4);
    if(wav_np>64 || wav_ns>64) fail("invalid asset prompt dimensions");
    size_t offset=8;
    for(uint32_t i=0;i<wav_np+wav_ns;i++) {
        uint32_t id=u32(wav_asset+offset);offset+=4;
        if(id>=VOC || id==151646) fail("invalid asset prompt token");
        if(i<wav_np) wav_prefix[i]=id;else wav_suffix[i-wav_np]=id;
    }
    if(offset+400*4+201*128*4+(VOC+1)*4>size) fail("truncated WAV assets");
    memcpy(wav_window,wav_asset+offset,400*4);offset+=400*4;
    memcpy(wav_filters,wav_asset+offset,201*128*4);offset+=201*128*4;
    finite_array(wav_window,400);finite_array(wav_filters,201*128);
    wav_offsets=wav_asset+offset;offset+=(VOC+1)*4;wav_bytes=wav_asset+offset;wav_bytes_size=(size_t)size-offset;
    uint32_t prev=0;
    for(int i=0;i<=VOC;i++) {uint32_t next=u32(wav_offsets+i*4);if(next<prev || next>wav_bytes_size || (!i && next)) fail("invalid token byte offsets");prev=next;}
    if(prev!=wav_bytes_size) fail("trailing token bytes");
    prof_set(previous);
}
static uint32_t wav_u16(const unsigned char *p) {return p[0]|((uint32_t)p[1]<<8);}
static float *wav_read(const char *path,int *samples) {
    FILE *f=open_file(path,"rb");unsigned char h[12],fmt[16];read_exact(f,h,12);
    if(memcmp(h,"RIFF",4) || memcmp(h+8,"WAVE",4)) fail("expected RIFF WAVE");
    uint64_t end=(uint64_t)u32(h+4)+8,pos=12,dataoff=0,bytes=0;int gotfmt=0;
    if(end<12 || end>LONG_MAX || fseek(f,0,SEEK_END) || ftell(f)!=(long)end) fail("truncated/invalid RIFF length");
    while(pos<end) {
        if(end-pos<8) fail("truncated RIFF chunk header");
        seek_to(f,pos);read_exact(f,h,8);uint64_t size=u32(h+4),next=pos+8+size+(size&1);
        if(next>end) fail("truncated RIFF chunk/padding");
        if(!memcmp(h,"fmt ",4)) {if(gotfmt || size<16) fail("invalid WAV fmt");read_exact(f,fmt,16);gotfmt=1;}
        if(!memcmp(h,"data",4)) {if(dataoff) fail("multiple WAV data chunks unsupported");dataoff=pos+8;bytes=size;}
        pos=next;
    }
    if(!gotfmt || !dataoff) fail("WAV missing fmt/data");
    uint32_t channels=wav_u16(fmt+2),rate=u32(fmt+4),align=wav_u16(fmt+12);
    if(wav_u16(fmt)!=1 || wav_u16(fmt+14)!=16) fail("only PCM16 WAV supported");
    if(!channels || channels>64 || !rate || rate>768000 || align!=channels*2 || u32(fmt+8)!=(uint64_t)rate*align || bytes%align) fail("invalid PCM16 WAV dimensions");
    uint64_t frames=bytes/align;
    if(!frames) fail("empty WAV");
    /* 精确整数商余实现 Python round 的 ties-to-even，随后截断到三十秒。 */
    uint64_t numerator=frames*16000,rounded=numerator/rate,remainder=numerator%rate;
    if(remainder*2>rate || (remainder*2==rate && (rounded&1))) rounded++;
    if(!rounded) rounded=1;
    uint32_t n=(uint32_t)(rounded>480000?480000:rounded);
    uint64_t needed=(uint64_t)ceil((double)(n-1)*frames/rounded)+2;if(needed>frames) needed=frames;
    float *mono=alloc((size_t)needed*4);seek_to(f,dataoff);
    unsigned char frame[128];
    for(uint64_t i=0;i<needed;i++) {
        read_exact(f,frame,align);float sum=0;
        for(uint32_t c=0;c<channels;c++) {uint32_t u=wav_u16(frame+2*c);int value=u>=32768?(int)u-65536:(int)u;sum+=(float)value/32768.0f;}
        mono[i]=sum/channels;
    }
    fclose(f);float *audio=alloc((size_t)n*4);
    for(uint32_t i=0;i<n;i++) {
        double x=rate==16000?(double)i:(double)i*frames/rounded;uint64_t j=(uint64_t)x;
        if(j>=frames-1) audio[i]=mono[frames-1];
        else audio[i]=(float)((double)mono[j]+(x-j)*((double)mono[j+1]-mono[j]));
    }
    free(mono);*samples=(int)n;return audio;
}
static float *wav_features(const char *path,uint32_t *length,uint32_t *count) {
    int previous=prof_set(P_FEATURES);
    int n;float *audio=wav_read(path,&n);
    /* Torch reflect padding 要求输入长于半窗；与 prepare 的实际拒绝边界一致。 */
    if(n<=200) fail("WAV needs more than 200 resampled samples for reflect STFT");
    int frames=n/160;*length=(uint32_t)frames;*count=(uint32_t)(((n/160+1)/2)/4);if(!*count) *count=1;
    float *mel=alloc(128*3000*4);memset(mel,0,128*3000*4);
    double *co=alloc(201*400*sizeof(double)),*si=alloc(201*400*sizeof(double));
    for(int k=0;k<=200;k++) for(int j=0;j<400;j++) {double a=2.0*3.14159265358979323846*k*j/400;co[k*400+j]=cos(a);si[k*400+j]=sin(a);}
    float maximum=-FLT_MAX;
    for(int t=0;t<frames;t++) {
        float windowed[400],power[201];
        for(int j=0;j<400;j++) {int ix=t*160+j-200;if(ix<0) ix=-ix;if(ix>=n) ix=2*n-2-ix;windowed[j]=audio[ix]*wav_window[j];}
        for(int k=0;k<=200;k++) {
            double re=0,im=0;for(int j=0;j<400;j++) {re+=windowed[j]*co[k*400+j];im-=windowed[j]*si[k*400+j];}
            float r=(float)re,s=(float)im;power[k]=r*r+s*s;
        }
        for(int m=0;m<128;m++) {
            float sum=0;for(int k=0;k<=200;k++) sum+=wav_filters[k*128+m]*power[k];
            float value=log10f(fmaxf(sum,1e-10f));mel[m*3000+t]=value;if(value>maximum) maximum=value;
        }
    }
    for(int m=0;m<128;m++) for(int t=0;t<frames;t++) mel[m*3000+t]=(fmaxf(mel[m*3000+t],maximum-8)+4)/4;
    free(audio);free(co);free(si);prof_set(previous);return mel;
}
/* Unicode White_Space 加 Python 额外的 ASCII 分隔符，与 re \s 一致。 */
static int text_space(uint32_t c) {return (c>=9 && c<=13) || (c>=28 && c<=32) || c==0x85 || c==0xa0 || c==0x1680 || (c>=0x2000 && c<=0x200a) || c==0x2028 || c==0x2029 || c==0x202f || c==0x205f || c==0x3000;}
static size_t text_utf8(const unsigned char *s,size_t n,uint32_t *cp) {
    unsigned char a=s[0];if(a<128) {*cp=a;return 1;}
    int count=a>=0xc2 && a<=0xdf?2:a>=0xe0 && a<=0xef?3:a>=0xf0 && a<=0xf4?4:0;
    if(!count) {*cp=0xfffd;return 1;}
    uint32_t c=a&((1u<<(7-count))-1);size_t j=1;
    for(;j<(size_t)count;j++) {
        if(j>=n || s[j]<0x80 || s[j]>0xbf || (j==1 && ((a==0xe0 && s[j]<0xa0) || (a==0xed && s[j]>0x9f) || (a==0xf0 && s[j]<0x90) || (a==0xf4 && s[j]>0x8f)))) {*cp=0xfffd;return j;}
        c=(c<<6)|(s[j]&63);
    }
    *cp=c;return j;
}
static size_t wav_text_size;
static char *wav_text(const uint32_t *ids,size_t count) {
    size_t size=0;
    for(size_t i=0;i<count;i++) {if(ids[i]>=VOC) fail("decode token out of range");size+=u32(wav_offsets+(ids[i]+1)*4)-u32(wav_offsets+ids[i]*4);}
    unsigned char *raw=alloc(size+1);size_t p=0;
    for(size_t i=0;i<count;i++) {uint32_t start=u32(wav_offsets+ids[i]*4),end=u32(wav_offsets+(ids[i]+1)*4);memcpy(raw+p,wav_bytes+start,end-start);p+=end-start;}
    /* 跨 token 拼接后才替换损坏 UTF-8，避免把半个汉字当作错误。 */
    char *text=alloc(size*3+1);size_t out=0;
    for(size_t i=0;i<size;) {uint32_t cp;size_t step=text_utf8(raw+i,size-i,&cp);if(cp==0xfffd) {memcpy(text+out,"\xef\xbf\xbd",3);out+=3;}else if(cp==0) {text[out++]=(char)0xff;}else {memcpy(text+out,raw+i,step);out+=step;}i+=step;}
    text[out]=0;free(raw);
    const char *stops[]={"<|user|>","<|assistant|>","<|im_end|>"};
    for(int i=0;i<3;i++) {char *at=strstr(text,stops[i]);if(at) *at=0;}
    char *begin=text;const char *starts[]={"<|text|>","<asr_text>"};
    for(int i=0;i<2;i++) {char *at=strstr(begin,starts[i]);if(at) begin=at+strlen(starts[i]);}
    /* 先归一化空白，再应用与 Python 相同的 language 前缀规则。 */
    char *clean=alloc(strlen(begin)+1);out=0;int pending=0;
    for(size_t i=0,len=strlen(begin);i<len;) {
        uint32_t cp;size_t step=text_utf8((unsigned char *)begin+i,len-i,&cp);
        if(text_space(cp)) pending=out!=0;
        else {if(pending) clean[out++]=' ';pending=0;memcpy(clean+out,begin+i,step);out+=step;}
        i+=step;
    }
    if(pending) clean[out++]=' ';
    clean[out]=0;begin=clean;
    if(!strncmp(begin,"language ",9)) {char *at=begin+9,*first=at;while((*at>='A' && *at<='Z') || (*at>='a' && *at<='z')) at++;if(at>first && *at==' ') begin=at+1;}
    out=0;
    for(size_t i=0;begin[i];) {
        if(!strncmp(begin+i,"<|",2)) {char *end=strchr(begin+i+2,'>');if(end && end>begin+i+3 && end[-1]=='|') {i=(size_t)(end-begin)+1;continue;}}
        clean[out++]=begin[i++];
    }
    clean[out]=0;
    /* 删除特殊标记后再次折叠、去掉开头标点。 */
    size_t i=0;while(clean[i] && strchr(" ,.;:!?-",clean[i])) i++;out=0;pending=0;
    for(;clean[i];i++) {if(clean[i]==' ') pending=out!=0;else {if(pending) clean[out++]=' ';pending=0;clean[out++]=clean[i];}}
    clean[out]=0;wav_text_size=out;
    /* 0xff 在有效 UTF-8 中不出现，临时替代嵌入 NUL 以便标记处理。 */
    for(size_t j=0;j<out;j++) if((unsigned char)clean[j]==0xff) clean[j]=0;
    free(text);return clean;
}

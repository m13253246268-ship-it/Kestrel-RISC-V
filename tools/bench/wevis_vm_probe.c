/* Build on K1 with the product includes, -O2 -march=rv64gcv -D_GNU_SOURCE
 * -ffp-contract=off -fopenmp -ffunction-sections -fdata-sections -Wl,--gc-sections -lm. */
#include "../../src/model/vllm_safetensors.c"

static uint32_t state = 42;
static uint32_t random_word(void) { return state = state * 1664525u + 1013904223u; }
static int compare_ns(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}
static int probe(int rows, int cols, FILE *model, const char *section) {
    const int nb = cols/32, tiles = rows/32;
    size_t bytes = (size_t)tiles * nb * 576;
    uint8_t *w = malloc(bytes), *wi = malloc(bytes);
    int8_t x[cols]; int32_t sx[nb], corr[nb]; float xd[nb], out[3][rows];
    if (!w || !wi) exit(2);
    for (size_t i=0; i<bytes; i+=576) {
        for (int r=0; r<32; ++r) {
            uint16_t sc = 0x2400 + (random_word() >> 19);
            memcpy(w+i+2*r, &sc, 2);
        }
        for (int j=64; j<576; ++j) w[i+j] = random_word() >> 24;
    }
    if (model) {
        VQFHeader hdr; VQFTensor tensor;
        rewind(model);
        if (fread(&hdr, sizeof hdr, 1, model)!=1 || hdr.magic!=VQF_MAGIC) exit(2);
        int found=0;
        for (unsigned i=0; i<hdr.n_tensors; ++i) {
            if (fread(&tensor,sizeof tensor,1,model)!=1) exit(2);
            if (strncmp(section,tensor.name,sizeof tensor.name)) continue;
            if (tensor.layout!=VQF_LAYOUT_Q4_COL || tensor.bytes!=bytes*hdr.arch.n_layers ||
                fseeko(model,tensor.offset,SEEK_SET) || fread(w,1,bytes,model)!=bytes) exit(2);
            found=1; break;
        }
        if (!found) exit(2);
    }
    rv_ime_rp_ctx rp = {w, wi, nb};
    for (int t=0; t<tiles; ++t) rv_ime_rp_worker(&rp,t);
    for (int b=0; b<nb; ++b) {
        xd[b] = (1+(random_word() >> 16))*0.0000013f; sx[b]=0;
        for (int j=0;j<32;++j) sx[b] += (x[b*32+j]=(int)(random_word()>>24)-128);
        corr[b]=8*sx[b];
    }
    long ns[3][9];
    for (int rep=-1; rep<9; ++rep) for (int pass=0; pass<3; ++pass) {
        int m=(pass+rep+1)%3;
        long start=rv_tp_ns();
        for (int t=0;t<tiles;++t) {
            float *o=out[m]+32*t;
            if (!m) rv_q4_dec_tile(w+(size_t)t*nb*576,x,xd,sx,o,NULL,nb,1);
            else for (int h=0;h<2;++h) {
                const uint8_t *p=wi+(size_t)(2*t+h)*nb*288;
                if (m==1) rv_vm_tile16_impl(p,x,xd,corr,nb,o+16*h,0);
                else rv_vm_tile16_impl(p,x,xd,corr,nb,o+16*h,1);
            }
        }
        if (rep>=0) ns[m][rep]=rv_tp_ns()-start;
    }
    int bad=0;
    printf("rows=%d K=%d",rows,cols);
    for(int m=0;m<3;++m) {
        int diff=0;
        for(int r=0;r<rows;++r) diff+=memcmp(out[m]+r,out[0]+r,sizeof(float))!=0;
        qsort(ns[m],9,sizeof(long),compare_ns); printf(" %s=%.3fms diff=%d",(const char*[]){"rvv","ime","k8"}[m],ns[m][4]*1e-6,diff); bad+=diff;
    }
    puts(""); free(w); free(wi); return bad;
}
int main(int argc, char **argv) {
    if (argc > 2 || !rv_vm_pin(0)) return 2;
    FILE *model = argc == 2 ? fopen(argv[1], "rb") : NULL;
    if (argc == 2 && !model) return 2;
    int bad = probe(2048,2048,model,"q4_q");
    bad += probe(1024,2048,model,"q4_k");
    bad += probe(6144,2048,model,"q4_gate");
    bad += probe(2048,6144,model,"q4_down");
    if (model) fclose(model);
    return !!bad;
}

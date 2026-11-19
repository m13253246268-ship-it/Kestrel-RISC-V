#include <stdlib.h>
#include <string.h>
#include "c-api.h"

typedef struct {
    const SherpaOnnxKeywordSpotter *model;
    const SherpaOnnxOnlineStream *stream;
} VoiceKws;

void voice_close(VoiceKws *v) {
    if (!v) return;
    if (v->stream) SherpaOnnxDestroyOnlineStream(v->stream);
    if (v->model) SherpaOnnxDestroyKeywordSpotter(v->model);
    free(v);
}
VoiceKws *voice_open(const char *encoder, const char *decoder,
                     const char *joiner, const char *tokens, const char *keywords) {
    SherpaOnnxKeywordSpotterConfig c;
    memset(&c, 0, sizeof(c));
    c.model_config.transducer.encoder = encoder;
    c.model_config.transducer.decoder = decoder;
    c.model_config.transducer.joiner = joiner;
    c.model_config.tokens = tokens;
    c.keywords_file = keywords;
    c.model_config.num_threads = 1;
    c.model_config.provider = "cpu";
    c.feat_config.sample_rate = 16000;
    c.feat_config.feature_dim = 80;
    c.max_active_paths = 4;
    c.num_trailing_blanks = 1;
    c.keywords_score = 1;
    c.keywords_threshold = 0.25f;
    VoiceKws *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->model = SherpaOnnxCreateKeywordSpotter(&c);
    if (!v->model) { voice_close(v); return NULL; }
    v->stream = SherpaOnnxCreateKeywordStream(v->model);
    if (!v->stream) { voice_close(v); return NULL; }
    return v;
}
int voice_reset(VoiceKws *v) {
    SherpaOnnxDestroyOnlineStream(v->stream);
    v->stream = SherpaOnnxCreateKeywordStream(v->model);
    return v->stream ? 0 : -1;
}
int voice_feed(VoiceKws *v, const float *samples, int n, char *keyword, int size) {
    SherpaOnnxOnlineStreamAcceptWaveform(v->stream, 16000, samples, n);
    while (SherpaOnnxIsKeywordStreamReady(v->model, v->stream)) {
        SherpaOnnxDecodeKeywordStream(v->model, v->stream);
        const SherpaOnnxKeywordResult *r = SherpaOnnxGetKeywordResult(v->model, v->stream);
        int hit = r && r->keyword && r->keyword[0];
        if (hit) {
            size_t len = strlen(r->keyword);
            if (len >= (size_t)size) { SherpaOnnxDestroyKeywordResult(r); return -1; }
            memcpy(keyword, r->keyword, len + 1);
        }
        SherpaOnnxDestroyKeywordResult(r);
        if (hit) return 1;
    }
    return 0;
}

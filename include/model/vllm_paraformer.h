#ifndef VLLM_PARAFORMER_H
#define VLLM_PARAFORMER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vllm_paraformer_model vllm_paraformer_model;
typedef struct vllm_paraformer_session vllm_paraformer_session;
enum { VLLM_PF_OK=0, VLLM_PF_INVALID=-1, VLLM_PF_ERROR=-2, VLLM_PF_STATE=-3 };
/* Borrowed arrays valid only during callback. Includes special token IDs.
 * Do not reenter the same session from its callback. logits are row-major
 * [count,8404], before log-softmax, and may be NULL when count is zero. */
typedef void (*vllm_paraformer_output)(void *user, const int *ids, size_t count, const float *logits);
/* FP32 VQF v7 only; fixed chunk [0,10,5], encoder/decoder lookback 4/1.
 * Errors are returned without exiting. err may be NULL when capacity is zero. */
int vllm_paraformer_model_create(const char *path, vllm_paraformer_model **out, char *err, size_t capacity);
/* Sessions retain the model; destroying the caller's model handle is safe. */
void vllm_paraformer_model_destroy(vllm_paraformer_model *model);
int vllm_paraformer_session_create(vllm_paraformer_model *model, vllm_paraformer_output output, void *user, vllm_paraformer_session **out);
/* Mono 16 kHz signed PCM16. NULL is permitted only for zero samples.
 * Arbitrary feed boundaries do not affect network chunks. */
int vllm_paraformer_session_feed(vllm_paraformer_session *session, const int16_t *pcm, size_t samples);
/* Flush once; feed/finish thereafter require reset. An inference error also
 * requires reset. Empty/<960-sample utterances follow FunASR's tail path. */
int vllm_paraformer_session_finish(vllm_paraformer_session *session);
int vllm_paraformer_session_reset(vllm_paraformer_session *session);
const char *vllm_paraformer_session_error(const vllm_paraformer_session *session);
void vllm_paraformer_session_destroy(vllm_paraformer_session *session);
/* Build src/model/vllm_paraformer.c with VLLM_PARAFORMER_NO_MAIN for a library.
 * Distinct sessions may be interleaved; concurrent access to handles requires
 * caller synchronization (including model retain/release). */
#ifdef __cplusplus
}
#endif
#endif

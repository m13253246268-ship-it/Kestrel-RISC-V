/* Audio8 ONNX bundle exporter. The wire format follows tools/asr/onnx_to_vqf.py. */
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include "vqf_asr.h"
#include "vllm_crypto.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
/* Cleanup is explicit; do not ask Windows SEH to unwind optimized C frames. */
#ifdef __MINGW32__
#define __USE_MINGW_SETJMP_NON_SEH
#endif
#include <setjmp.h>
#include <limits.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#define seek64 _fseeki64
#define tell64 _ftelli64
#else
#include <sys/types.h>
#define seek64 fseeko
#define tell64 ftello
#endif

typedef struct Mem { struct Mem *next; } Mem;
typedef struct { const unsigned char *p; size_t n; } Slice;
typedef struct { char *p; size_t n, cap; } Str;
typedef struct { uint64_t dims[4], off, size; unsigned nd, type; unsigned char hash[32]; } Array;
typedef struct {
    jmp_buf fail;
    Mem *mem;
    FILE *spool, *output, *input;
    Array *arrays;
    size_t count, capacity, shared;
    uint64_t end;
    const char *error;
} Ctx;
typedef struct { unsigned id, wire; uint64_t v; Slice bytes; } Field;

static void die(Ctx *c, const char *s) { c->error = s; longjmp(c->fail, 1); }
static void *alloc(Ctx *c, size_t n) {
    Mem *m;
    if (n > SIZE_MAX - sizeof(*m)) die(c, "allocation overflow");
    m = malloc(sizeof(*m) + (n ? n : 1));
    if (!m) die(c, "out of memory");
    m->next = c->mem; c->mem = m;
    return m + 1;
}
static void release(Ctx *c, void *p) {
    Mem **m = &c->mem;
    while (*m && (void *)(*m + 1) != p) m = &(*m)->next;
    if (*m) { Mem *old = *m; *m = old->next; free(old); }
}
static char *copy(Ctx *c, const void *p, size_t n) {
    char *s = alloc(c, n + 1); memcpy(s, p, n); s[n] = 0; return s;
}
static void append(Ctx *c, Str *s, const void *p, size_t n) {
    if (n > SIZE_MAX - s->n - 1) die(c, "string overflow");
    if (s->n + n + 1 > s->cap) {
        size_t cap = s->n + n + 1;
        char *q;
        if (cap <= SIZE_MAX / 2) cap *= 2;
        q = alloc(c, cap);
        if (s->n) memcpy(q, s->p, s->n);
        release(c, s->p); s->p = q; s->cap = cap;
    }
    memcpy(s->p + s->n, p, n); s->n += n; s->p[s->n] = 0;
}
static void text(Ctx *c, Str *s, const char *p) { append(c, s, p, strlen(p)); }
static void number(Ctx *c, Str *s, uint64_t v) {
    char b[32]; snprintf(b, sizeof(b), "%llu", (unsigned long long)v); text(c, s, b);
}
static void quoted(Ctx *c, Str *s, Slice a) {
    size_t i; text(c, s, "\"");
    for (i = 0; i < a.n; i++) {
        unsigned ch = a.p[i]; char b[7];
        switch (ch) {
        case '"': text(c, s, "\\\""); break;
        case '\\': text(c, s, "\\\\"); break;
        case '\b': text(c, s, "\\b"); break;
        case '\f': text(c, s, "\\f"); break;
        case '\n': text(c, s, "\\n"); break;
        case '\r': text(c, s, "\\r"); break;
        case '\t': text(c, s, "\\t"); break;
        default:
            if (ch < 32) { snprintf(b, sizeof(b), "\\u%04x", ch); text(c, s, b); }
            else append(c, s, a.p + i, 1);
        }
    }
    text(c, s, "\"");
}
static Slice strslice(const char *s) { Slice a; a.p = (const unsigned char *)s; a.n = strlen(s); return a; }
static int equal(Slice a, const char *s) { return a.n == strlen(s) && !memcmp(a.p, s, a.n); }
static uint64_t le(const unsigned char *p, unsigned n) {
    uint64_t v = 0; unsigned i; for (i = 0; i < n; i++) v |= (uint64_t)p[i] << (8 * i); return v;
}
static void put(unsigned char *p, uint64_t v, unsigned n) { unsigned i; for (i = 0; i < n; i++) p[i] = (unsigned char)(v >> (i * 8)); }
#ifdef _WIN32
static wchar_t *wide(Ctx *c, const char *s) {
    UINT cp = CP_UTF8; int n = MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    wchar_t *w;
    if (!n) { cp = CP_ACP; n = MultiByteToWideChar(cp, 0, s, -1, NULL, 0); }
    if (!n) die(c, "invalid filename encoding");
    w = alloc(c, (size_t)n * sizeof(*w));
    if (!MultiByteToWideChar(cp, cp == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0, s, -1, w, n)) die(c, "filename conversion failed");
    return w;
}
#endif
static FILE *openfile(Ctx *c, const char *s, int write_mode) {
#ifdef _WIN32
    wchar_t *w = wide(c, s); FILE *f = _wfopen(w, write_mode ? L"w+b" : L"rb"); release(c, w); return f;
#else
    (void)c; return fopen(s, write_mode ? "w+b" : "rb");
#endif
}
static char *pathjoin(Ctx *c, const char *a, const char *b) { Str s = {0}; text(c, &s, a); text(c, &s, "/"); text(c, &s, b); return s.p; }
static Slice readfile(Ctx *c, const char *path) {
    Slice s; int64_t n;
    c->input = openfile(c, path, 0);
    if (!c->input) { fprintf(stderr, "[asr] cannot open %s\n", path); die(c, "input open failed"); }
    if (seek64(c->input, 0, SEEK_END) || (n = tell64(c->input)) < 0 || (uint64_t)n > SIZE_MAX - 1 || seek64(c->input, 0, SEEK_SET)) die(c, "input size failed");
    s.n = (size_t)n; s.p = alloc(c, s.n + 1);
    if (fread((void *)s.p, 1, s.n, c->input) != s.n) die(c, "input read failed");
    ((unsigned char *)s.p)[s.n] = 0; fclose(c->input); c->input = NULL; return s;
}
static uint64_t varint(Ctx *c, Slice *s) {
    uint64_t v = 0; unsigned i;
    for (i = 0; i < 10; i++) {
        unsigned b;
        if (!s->n) die(c, "truncated protobuf varint");
        b = *s->p++; s->n--;
        if (i == 9 && b > 1) die(c, "protobuf varint overflow");
        v |= (uint64_t)(b & 127) << (7 * i);
        if (!(b & 128)) return v;
    }
    die(c, "invalid protobuf varint"); return 0;
}
static int field(Ctx *c, Slice *s, Field *f) {
    uint64_t tag, n;
    if (!s->n) return 0;
    tag = varint(c, s);
    if (!(tag >> 3) || (tag >> 3) > 0x1fffffff) die(c, "invalid protobuf tag");
    memset(f, 0, sizeof(*f)); f->id = (unsigned)(tag >> 3); f->wire = (unsigned)(tag & 7);
    if (!f->wire) { f->v = varint(c, s); return 1; }
    if (f->wire == 2) n = varint(c, s);
    else if (f->wire == 1) n = 8;
    else if (f->wire == 5) n = 4;
    else { die(c, "unsupported protobuf wire type"); return 0; }
    if (n > s->n) die(c, "truncated protobuf field");
    f->bytes.p = s->p; f->bytes.n = (size_t)n;
    if (f->wire != 2) f->v = le(s->p, (unsigned)n);
    s->p += (size_t)n; s->n -= (size_t)n; return 1;
}
static Slice get(Ctx *c, Slice s, unsigned id) {
    Field f; Slice empty = {(const unsigned char *)"", 0};
    while (field(c, &s, &f)) if (f.id == id) { if (f.wire != 2) die(c, "expected protobuf string/message"); return f.bytes; }
    return empty;
}
static void wire(Ctx *c, Field f, unsigned w) { if (f.wire != w) die(c, "wrong protobuf field wire type"); }

/* Validate JSON while locating values; preserve the reference bundle's number spelling. */
static void ws(Slice *s) { while (s->n && (*s->p == ' ' || *s->p == '\n' || *s->p == '\r' || *s->p == '\t')) { s->p++; s->n--; } }
static void take(Ctx *c, Slice *s, unsigned ch) { if (!s->n || *s->p != ch) die(c, "invalid JSON"); s->p++; s->n--; }
static Slice jstring(Ctx *c, Slice *s) {
    const unsigned char *start = s->p; take(c, s, '"');
    while (s->n && *s->p != '"') {
        unsigned b = *s->p++; s->n--;
        if (b < 32) die(c, "JSON control character");
        if (b == '\\') {
            unsigned i;
            if (!s->n) die(c, "truncated JSON escape");
            b = *s->p++; s->n--;
            if (b == 'u') {
                for (i = 0; i < 4; i++) {
                    if (!s->n || !((*s->p >= '0' && *s->p <= '9') || (*s->p >= 'a' && *s->p <= 'f') || (*s->p >= 'A' && *s->p <= 'F'))) die(c, "invalid JSON unicode escape");
                    s->p++; s->n--;
                }
            } else if (!strchr("\"\\/bfnrt", (int)b)) die(c, "invalid JSON escape");
        }
    }
    take(c, s, '"'); { Slice r = {start, (size_t)(s->p - start)}; return r; }
}
static void jvalue(Ctx *c, Slice *s, unsigned depth) {
    unsigned ch; ws(s);
    if (!s->n || depth > 128) die(c, "invalid JSON depth/value");
    ch = *s->p;
    if (ch == '"') { (void)jstring(c, s); return; }
    if (ch == '{' || ch == '[') {
        unsigned close = ch == '{' ? '}' : ']'; s->p++; s->n--; ws(s);
        if (s->n && *s->p == close) { s->p++; s->n--; return; }
        for (;;) {
            if (ch == '{') { (void)jstring(c, s); ws(s); take(c, s, ':'); }
            jvalue(c, s, depth + 1); ws(s);
            if (s->n && *s->p == close) { s->p++; s->n--; return; }
            take(c, s, ','); ws(s);
        }
    }
    if (ch == 't' || ch == 'f' || ch == 'n') {
        const char *lit = ch == 't' ? "true" : ch == 'f' ? "false" : "null"; size_t n = strlen(lit);
        if (s->n < n || memcmp(s->p, lit, n)) die(c, "invalid JSON literal");
        s->p += n; s->n -= n; return;
    }
    if (ch == '-') { s->p++; s->n--; }
    if (!s->n || *s->p < '0' || *s->p > '9') die(c, "invalid JSON number");
    if (*s->p == '0') { s->p++; s->n--; }
    else while (s->n && *s->p >= '0' && *s->p <= '9') { s->p++; s->n--; }
    if (s->n && *s->p == '.') {
        s->p++; s->n--;
        if (!s->n || *s->p < '0' || *s->p > '9') die(c, "invalid JSON fraction");
        while (s->n && *s->p >= '0' && *s->p <= '9') { s->p++; s->n--; }
    }
    if (s->n && (*s->p == 'e' || *s->p == 'E')) {
        s->p++; s->n--; if (s->n && (*s->p == '+' || *s->p == '-')) { s->p++; s->n--; }
        if (!s->n || *s->p < '0' || *s->p > '9') die(c, "invalid JSON exponent");
        while (s->n && *s->p >= '0' && *s->p <= '9') { s->p++; s->n--; }
    }
}
static char *minify(Ctx *c, Slice s) {
    Slice check = s; Str r = {0}; jvalue(c, &check, 0); ws(&check); if (check.n) die(c, "JSON trailing bytes");
    while (s.n) {
        ws(&s); if (!s.n) break;
        if (*s.p == '"') { Slice v = jstring(c, &s); append(c, &r, v.p, v.n); }
        else { append(c, &r, s.p, 1); s.p++; s.n--; }
    }
    return r.p;
}
static Slice jget(Ctx *c, Slice s, const char *key) {
    ws(&s); take(c, &s, '{'); ws(&s);
    while (s.n && *s.p != '}') {
        Slice k = jstring(c, &s), v; const unsigned char *start;
        ws(&s); take(c, &s, ':'); ws(&s); start = s.p; jvalue(c, &s, 0); v.p = start; v.n = (size_t)(s.p - start);
        if (k.n == strlen(key) + 2 && !memcmp(k.p + 1, key, k.n - 2)) return v;
        ws(&s); if (s.n && *s.p == '}') break; take(c, &s, ','); ws(&s);
    }
    die(c, "missing JSON key"); { Slice z = {NULL, 0}; return z; }
}
static char *jpath(Ctx *c, Slice s) {
    Slice v = jstring(c, &s); size_t i;
    for (i = 1; i + 1 < v.n; i++) if (v.p[i] == '\\' || v.p[i] == 0) die(c, "escaped asset paths are unsupported");
    if (s.n) die(c, "invalid asset path");
    return copy(c, v.p + 1, v.n - 2);
}
static char *jsonfile(Ctx *c, const char *dir, const char *name) {
    Slice raw = readfile(c, pathjoin(c, dir, name)); char *s = minify(c, raw); release(c, (void *)raw.p); return s;
}
static void writebytes(Ctx *c, FILE *f, const void *p, size_t n) { if (fwrite(p, 1, n, f) != n) die(c, "output write failed"); }
static void seekto(Ctx *c, FILE *f, uint64_t off) { if (off > INT64_MAX || seek64(f, (int64_t)off, SEEK_SET)) die(c, "file seek failed"); }
static void zeros(Ctx *c, FILE *f, uint64_t n) { unsigned char b[4096] = {0}; while (n) { size_t k = n > sizeof(b) ? sizeof(b) : (size_t)n; writebytes(c, f, b, k); n -= k; } }
static const char *dtype(Ctx *c, unsigned type) {
    switch (type) { case 1: return "<f4"; case 6: return "<i4"; case 7: return "<i8"; case 9: return "|b1"; default: die(c, "unsupported tensor dtype"); return ""; }
}
static unsigned itemsize(Ctx *c, unsigned type) { (void)dtype(c, type); return type == 7 ? 8 : type == 9 ? 1 : 4; }
static char *arrayname(Ctx *c, size_t i) { char s[32]; snprintf(s, sizeof(s), "t%06llu", (unsigned long long)i); return copy(c, s, strlen(s)); }
static char *add(Ctx *c, Array a, Slice raw) {
    size_t i; uint64_t size = itemsize(c, a.type); unsigned k; unsigned char buf[65536];
    for (k = 0; k < a.nd; k++) { if (a.dims[k] > UINT32_MAX || (a.dims[k] && size > SIZE_MAX / a.dims[k])) die(c, "tensor size overflow"); size *= a.dims[k]; }
    if (size != raw.n) die(c, "tensor data length does not match shape");
    a.size = size; vc_sm3(raw.p, raw.n, a.hash);
    for (i = 0; i < c->count; i++) {
        Array *b = &c->arrays[i]; uint64_t pos;
        if (a.type != b->type || a.nd != b->nd || a.size != b->size || memcmp(a.dims, b->dims, sizeof(a.dims)) || memcmp(a.hash, b->hash, 32)) continue;
        if (fflush(c->spool)) die(c, "spool flush failed");
        seekto(c, c->spool, b->off);
        for (pos = 0; pos < size;) {
            size_t n = size - pos > sizeof(buf) ? sizeof(buf) : (size_t)(size - pos);
            if (fread(buf, 1, n, c->spool) != n || memcmp(buf, raw.p + (size_t)pos, n)) die(c, "tensor digest collision or spool read failure");
            pos += n;
        }
        c->shared++; return arrayname(c, i);
    }
    if (c->count == c->capacity) {
        Array *p; size_t cap = c->capacity ? c->capacity * 2 : 512;
        if (cap > SIZE_MAX / sizeof(*p)) die(c, "too many arrays");
        p = alloc(c, cap * sizeof(*p)); if (c->count) memcpy(p, c->arrays, c->count * sizeof(*p)); release(c, c->arrays); c->arrays = p; c->capacity = cap;
    }
    a.off = (c->end + 63) & ~UINT64_C(63);
    seekto(c, c->spool, c->end); zeros(c, c->spool, a.off - c->end); writebytes(c, c->spool, raw.p, raw.n);
    c->end = a.off + size; c->arrays[c->count] = a; return arrayname(c, c->count++);
}
static char *tensor(Ctx *c, Slice s, int transpose) {
    Array a = {0}; Field f; Slice raw = {NULL, 0}; unsigned char *trans = NULL; char *name;
    while (field(c, &s, &f)) {
        if (f.id == 1) {
            Slice packed = f.bytes;
            if (f.wire != 0 && f.wire != 2) die(c, "invalid tensor dimensions");
            do {
                uint64_t v = f.wire == 0 ? f.v : varint(c, &packed);
                if (a.nd == 4 || v > INT64_MAX) die(c, "unsupported tensor shape");
                a.dims[a.nd++] = v;
            } while (f.wire == 2 && packed.n);
        } else if (f.id == 2) { wire(c, f, 0); a.type = (unsigned)f.v; }
        else if (f.id == 9) { wire(c, f, 2); raw = f.bytes; }
        else if (f.id == 13 || (f.id == 14 && f.v)) die(c, "external ONNX tensors are unsupported");
        else if (f.id == 4 || f.id == 5 || f.id == 6 || f.id == 7 || f.id == 10 || f.id == 11) die(c, "ONNX tensor requires raw_data");
    }
    if (!raw.p) die(c, "ONNX tensor missing raw_data");
    if (transpose) {
        uint64_t r, col, rows = a.dims[0], cols = a.dims[1];
        if (a.nd != 2 || a.type != 1 || rows > SIZE_MAX / 4 || (cols && rows * 4 > SIZE_MAX / cols) || rows * cols * 4 != raw.n) die(c, "invalid lm_head matrix");
        trans = alloc(c, raw.n);
        for (r = 0; r < rows; r++) for (col = 0; col < cols; col++) memcpy(trans + (size_t)(col * rows + r) * 4, raw.p + (size_t)(r * cols + col) * 4, 4);
        a.dims[0] = cols; a.dims[1] = rows; raw.p = trans;
    }
    name = add(c, a, raw); release(c, trans); return name;
}
static void pair(Ctx *c, Str *s, Slice key, const char *value) {
    if (s->n) { text(c, s, ","); } quoted(c, s, key); text(c, s, ":"); quoted(c, s, strslice(value));
}
static void floatjson(Ctx *c, Str *s, uint64_t bits) {
    uint32_t u = (uint32_t)bits; float f; char b[64]; memcpy(&f, &u, 4);
    if (!isfinite(f)) die(c, "nonfinite attribute unsupported");
    {
        int precision;
        for (precision = 1; precision <= 17; precision++) {
            snprintf(b, sizeof(b), "%.*g", precision, (double)f);
            if (strtod(b, NULL) == (double)f) break;
        }
    }
    text(c, s, b);
    if (!strchr(b, '.') && !strchr(b, 'e') && !strchr(b, 'E')) text(c, s, ".0");
}
static void intjson(Ctx *c, Str *s, uint64_t v) { char b[32]; int64_t i; memcpy(&i, &v, 8); snprintf(b, sizeof(b), "%lld", (long long)i); text(c, s, b); }
static char *attribute(Ctx *c, Slice a, char **tensor_ref) {
    Slice scan = a; Field f; uint64_t type = 0; Str val = {0}; int found = 0;
    *tensor_ref = NULL;
    while (field(c, &scan, &f)) if (f.id == 20) { wire(c, f, 0); type = f.v; }
    if (type == 6 || type == 7) text(c, &val, "[");
    while (field(c, &a, &f)) {
        if (type == 1 && f.id == 2) { wire(c, f, 5); floatjson(c, &val, f.v); found++; }
        else if (type == 2 && f.id == 3) { wire(c, f, 0); intjson(c, &val, f.v); found++; }
        else if (type == 3 && f.id == 4) { wire(c, f, 2); quoted(c, &val, f.bytes); found++; }
        else if (type == 4 && f.id == 5) { wire(c, f, 2); *tensor_ref = tensor(c, f.bytes, 0); text(c, &val, "{\"tensor\":"); quoted(c, &val, strslice(*tensor_ref)); text(c, &val, "}"); found++; }
        else if ((type == 6 && f.id == 7) || (type == 7 && f.id == 8)) {
            Slice packed = f.bytes;
            if (f.wire != 2 && f.wire != (type == 6 ? 5u : 0u)) die(c, "invalid repeated attribute");
            while (f.wire != 2 || packed.n) {
                uint64_t v = f.v;
                if (f.wire == 2) {
                    if (type == 6) { if (packed.n < 4) die(c, "truncated packed float"); v = le(packed.p, 4); packed.p += 4; packed.n -= 4; }
                    else v = varint(c, &packed);
                }
                if (found++) text(c, &val, ",");
                if (type == 6) floatjson(c, &val, v); else intjson(c, &val, v);
                if (f.wire != 2) break;
            }
        }
    }
    if (type == 6 || type == 7) text(c, &val, "]");
    else if (!found || found != 1) die(c, "unsupported or invalid ONNX attribute");
    return val.p;
}
static char *strings(Ctx *c, Slice s, unsigned id, int names) {
    Str v = {0}; Field f; int first = 1; text(c, &v, "[");
    while (field(c, &s, &f)) if (f.id == id) {
        Slice value; wire(c, f, 2); value = names ? get(c, f.bytes, 1) : f.bytes;
        if (!first) { text(c, &v, ","); } first = 0; quoted(c, &v, value);
    }
    text(c, &v, "]"); return v.p;
}
static char *graph(Ctx *c, const char *dir, const char *path, char **head_ref) {
    Slice raw = readfile(c, pathjoin(c, dir, path)), g = get(c, raw, 7), scan = g, head = {NULL, 0};
    Field f; Str tensors = {0}, nodes = {0}, roles = {0}, result = {0}; size_t operators = 0;
    if (!g.n) die(c, "missing ONNX graph");
    *head_ref = NULL;
    while (field(c, &scan, &f)) if (f.id == 1) {
        Slice n, op, name, second = {NULL, 0}; Field x; unsigned inputs = 0; wire(c, f, 2); n = f.bytes;
        op = get(c, n, 4); name = get(c, n, 3);
        while (field(c, &n, &x)) if (x.id == 1) { wire(c, x, 2); if (++inputs == 2) second = x.bytes; }
        if (equal(op, "MatMul") || equal(op, "Gemm")) if (inputs > 1) { char *v = copy(c, second.p, second.n); pair(c, &roles, name, v); }
        if (equal(name, "/lm_head/MatMul")) { if (!second.p || head.p) die(c, "invalid lm_head"); head = second; }
    }
    scan = g;
    while (field(c, &scan, &f)) if (f.id == 5) {
        Slice name; char *ref; int trans; wire(c, f, 2); name = get(c, f.bytes, 8);
        trans = head.p && head.n == name.n && !memcmp(head.p, name.p, name.n); ref = tensor(c, f.bytes, trans);
        if (trans) {
            Str canonical = {0}; append(c, &canonical, name.p, name.n); text(c, &canonical, ".stored_transposed");
            pair(c, &tensors, strslice(canonical.p), ref); *head_ref = ref;
            if (nodes.n) text(c, &nodes, ",");
            text(c, &nodes, "{\"op\":\"Transpose\",\"name\":\"vqf_shared_lm_head\",\"inputs\":["); quoted(c, &nodes, strslice(canonical.p));
            text(c, &nodes, "],\"outputs\":["); quoted(c, &nodes, name); text(c, &nodes, "],\"attrs\":{\"perm\":[1,0]}}"); operators++;
        } else pair(c, &tensors, name, ref);
    }
    scan = g;
    while (field(c, &scan, &f)) if (f.id == 1) {
        Slice n = f.bytes, attrs_scan = n, op = get(c, n, 4), name = get(c, n, 3); Field x; Str attrs = {0}; char *constant = NULL;
        while (field(c, &attrs_scan, &x)) if (x.id == 5) {
            Slice key; char *ref, *v; wire(c, x, 2); key = get(c, x.bytes, 1); v = attribute(c, x.bytes, &ref);
            if (attrs.n) { text(c, &attrs, ","); } quoted(c, &attrs, key); text(c, &attrs, ":"); text(c, &attrs, v);
            if (equal(key, "value")) constant = ref;
        }
        if (equal(op, "Constant")) {
            Slice output = get(c, n, 2); if (!constant || !output.n) die(c, "unsupported Constant"); pair(c, &tensors, output, constant);
        } else {
            if (nodes.n) { text(c, &nodes, ","); } text(c, &nodes, "{\"op\":"); quoted(c, &nodes, op);
            text(c, &nodes, ",\"name\":"); quoted(c, &nodes, name); text(c, &nodes, ",\"inputs\":"); text(c, &nodes, strings(c, n, 1, 0));
            text(c, &nodes, ",\"outputs\":"); text(c, &nodes, strings(c, n, 2, 0)); text(c, &nodes, ",\"attrs\":{"); if (attrs.n) append(c, &nodes, attrs.p, attrs.n); text(c, &nodes, "}}"); operators++;
        }
    }
    text(c, &result, "{\"inputs\":"); text(c, &result, strings(c, g, 11, 1)); text(c, &result, ",\"outputs\":"); text(c, &result, strings(c, g, 12, 1));
    text(c, &result, ",\"tensors\":{"); if (tensors.n) append(c, &result, tensors.p, tensors.n);
    text(c, &result, "},\"nodes\":["); if (nodes.n) append(c, &result, nodes.p, nodes.n);
    text(c, &result, "],\"weight_roles\":{"); if (roles.n) append(c, &result, roles.p, roles.n); text(c, &result, "}}");
    release(c, (void *)raw.p); fprintf(stderr, "[asr] %s: %llu operators\n", path, (unsigned long long)operators); return result.p;
}
static char *npy(Ctx *c, Slice s) {
    Array a = {0}; size_t off, hn; char *header, *p, *end; unsigned size;
    if (s.n < 10 || memcmp(s.p, "\223NUMPY", 6) || s.p[6] != 1 || s.p[7] != 0) die(c, "unsupported NPY header (requires v1)");
    hn = (size_t)le(s.p + 8, 2); off = 10 + hn; if (off > s.n) die(c, "truncated NPY header");
    header = copy(c, s.p + 10, hn);
    p = strstr(header, "'descr':"); if (!p) die(c, "NPY missing dtype"); p += 8; while (*p == ' ') p++;
    if (!strncmp(p, "'<f4'", 5)) a.type = 1;
    else if (!strncmp(p, "'<i4'", 5)) a.type = 6;
    else if (!strncmp(p, "'<i8'", 5)) a.type = 7;
    else if (!strncmp(p, "'|b1'", 5)) a.type = 9;
    else die(c, "unsupported NPY dtype");
    p = strstr(header, "'fortran_order':"); if (!p) die(c, "NPY missing order"); p += 16; while (*p == ' ') p++;
    if (strncmp(p, "False", 5)) die(c, "Fortran-order NPY unsupported");
    p = strstr(header, "'shape':"); if (!p) die(c, "NPY missing shape"); p = strchr(p, '('); if (!p) die(c, "invalid NPY shape"); p++;
    for (;;) {
        unsigned long long v;
        while (*p == ' ') p++;
        if (*p == ')') break;
        if (*p < '0' || *p > '9' || a.nd == 4) die(c, "invalid NPY dimension");
        v = strtoull(p, &end, 10); if (v > UINT32_MAX || end == p) die(c, "NPY dimension overflow"); a.dims[a.nd++] = v; p = end;
        while (*p == ' ') p++;
        if (*p == ',') p++; else if (*p != ')') die(c, "invalid NPY shape separator");
    }
    size = itemsize(c, a.type); (void)size; s.p += off; s.n -= off; return add(c, a, s);
}
static char *projector(Ctx *c, const char *path) {
    Slice raw = readfile(c, path), s = raw; Str r = {0}; unsigned count = 0;
    while (s.n >= 4 && le(s.p, 4) == 0x04034b50) {
        size_t name_len, extra_len, off; uint64_t length; char *name, *ref;
        if (s.n < 30 || (le(s.p + 6, 2) & 9) || le(s.p + 8, 2)) die(c, "NPZ requires unencrypted stored entries");
        length = le(s.p + 18, 4); if (length != le(s.p + 22, 4)) die(c, "invalid stored ZIP sizes");
        name_len = (size_t)le(s.p + 26, 2); extra_len = (size_t)le(s.p + 28, 2); off = 30 + name_len + extra_len;
        if (off > s.n || name_len < 4 || memcmp(s.p + 30 + name_len - 4, ".npy", 4)) die(c, "invalid NPZ entry");
        if (length == UINT32_MAX) {
            size_t e = 30 + name_len, stop = off; int seen = 0;
            while (e + 4 <= stop) {
                size_t n = (size_t)le(s.p + e + 2, 2); unsigned tag = (unsigned)le(s.p + e, 2); e += 4;
                if (n > stop - e) die(c, "truncated ZIP extra field");
                if (tag == 1) { if (n < 16 || le(s.p + e, 8) != le(s.p + e + 8, 8)) die(c, "invalid ZIP64 sizes"); length = le(s.p + e, 8); seen = 1; }
                e += n;
            }
            if (!seen) die(c, "missing ZIP64 sizes");
        }
        if (length > s.n - off) die(c, "truncated NPZ entry");
        name = copy(c, s.p + 30, name_len - 4);
        { Slice data = {s.p + off, (size_t)length}; ref = npy(c, data); }
        pair(c, &r, strslice(name), ref); count++; s.p += off + (size_t)length; s.n -= off + (size_t)length;
    }
    if (!count || s.n < 4 || le(s.p, 4) != 0x02014b50) die(c, "missing NPZ directory");
    release(c, (void *)raw.p); return r.p;
}
static uint64_t fnv(uint64_t h, const unsigned char *p, size_t n) { size_t i; for (i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(1099511628211); return h; }
/* ASR-only qtype 0x100, layout 1: original row-major signed bytes,
 * followed by one little-endian FP32 positive scale per original row. */
static void quantize(Ctx *c) {
    FILE *old = c->spool;
    FILE *next = tmpfile();
    if (!next) die(c, "cannot create quantized spool");
    c->spool = next; c->input = old; c->end = 0;
    for (size_t i = 0; i < c->count; i++) {
        Array *a = &c->arrays[i];
        uint64_t off = (c->end + 63) & ~UINT64_C(63), size = a->size;
        unsigned char *raw = alloc(c, (size_t)size);
        seekto(c, old, a->off);
        if (fread(raw, 1, (size_t)size, old) != size) die(c, "quantization spool read failed");
        zeros(c, next, off - c->end);
        if (a->type == 1 && a->nd >= 2 && a->dims[0] > 1 && size / 4 / a->dims[0] > 1) {
            size_t rows = (size_t)a->dims[0], cols = (size_t)size / 4 / rows;
            float *src = (float *)raw, *scales = alloc(c, rows * 4);
            int8_t *q = alloc(c, rows * cols);
            for (size_t r = 0; r < rows; r++) {
                float peak = 0;
                for (size_t k = 0; k < cols; k++) {
                    float v = src[r * cols + k];
                    if (!isfinite(v)) die(c, "nonfinite quantization input");
                    if (fabsf(v) > peak) peak = fabsf(v);
                }
                float scale = peak ? peak / 127.0f : 1.0f;
                if (!isfinite(scale) || scale <= 0) die(c, "invalid quantization scale");
                scales[r] = scale;
                for (size_t k = 0; k < cols; k++) {
                    float v = roundf(src[r * cols + k] / scale);
                    q[r * cols + k] = (int8_t)fmaxf(-127.0f, fminf(127.0f, v));
                }
            }
            writebytes(c, next, q, rows * cols); writebytes(c, next, scales, rows * 4);
            size = rows * cols + rows * 4; a->type = 0x100;
            release(c, q); release(c, scales);
        } else writebytes(c, next, raw, (size_t)size);
        release(c, raw); a->off = off; a->size = size; c->end = off + size;
    }
    fclose(old); c->input = NULL;
}
static void finish(Ctx *c, Str *meta, const char *out) {
    size_t i, n = c->count + 1; uint64_t off = (c->end + 63) & ~UINT64_C(63), dataoff, filelen, h = UINT64_C(14695981039346656037), remaining;
    unsigned char header[448] = {0}, *directory, *buffer;
    text(c, meta, ",\"arrays\":{");
    for (i = 0; i < c->count; i++) {
        Array *a = &c->arrays[i]; unsigned k;
        if (i) { text(c, meta, ","); } quoted(c, meta, strslice(arrayname(c, i))); text(c, meta, ":{\"shape\":[");
        for (k = 0; k < a->nd; k++) { if (k) text(c, meta, ","); number(c, meta, a->dims[k]); }
        text(c, meta, "],\"dtype\":"); quoted(c, meta, strslice(a->type == 0x100 ? "|i1" : dtype(c, a->type)));
        if (a->type == 0x100) text(c, meta, ",\"quantization\":\"signed-row-f32-v1\"");
        text(c, meta, "}");
    }
    text(c, meta, "}}");
    seekto(c, c->spool, c->end); zeros(c, c->spool, off - c->end); writebytes(c, c->spool, meta->p, meta->n); c->end = off + meta->n;
    if (n > UINT32_MAX || n > (SIZE_MAX - 448) / 64 || meta->n > UINT32_MAX) die(c, "VQF directory overflow");
    dataoff = (448 + n * 64 + 4095) & ~UINT64_C(4095); filelen = dataoff + c->end;
    put(header, 0x57465146, 4); put(header + 4, 6, 4); put(header + 8, 1u << 12, 4); put(header + 12, n, 4);
    memcpy(header + 16, "ASR-NUMPY-v1", 12); put(header + 200, dataoff, 8); put(header + 208, filelen, 8);
    directory = alloc(c, n * 64); memset(directory, 0, n * 64);
    for (i = 0; i < n; i++) {
        unsigned char *d = directory + i * 64; uint64_t rows = 1, cols = 1; unsigned k;
        if (i == c->count) { memcpy(d, "asr_metadata", 12); put(d + 32, 6, 4); cols = meta->n; put(d + 48, dataoff + off, 8); put(d + 56, meta->n, 8); }
        else {
            Array *a = &c->arrays[i]; char *name = arrayname(c, i); memcpy(d, name, strlen(name)); put(d + 32, a->type == 0x100 ? 0x100 : a->type == 1 ? 1 : 6, 4);
            if (a->type == 0x100) put(d + 44, 1, 4);
            if (a->nd) rows = a->dims[0];
            for (k = 1; k < a->nd; k++) { if (a->dims[k] && cols > UINT32_MAX / a->dims[k]) die(c, "VQF column overflow"); cols *= a->dims[k]; }
            put(d + 48, dataoff + a->off, 8); put(d + 56, a->size, 8);
        }
        put(d + 36, rows, 4); put(d + 40, cols, 4);
    }
    c->output = openfile(c, out, 1); if (!c->output) die(c, "cannot open output");
    writebytes(c, c->output, header, sizeof(header)); writebytes(c, c->output, directory, n * 64); zeros(c, c->output, dataoff - 448 - n * 64);
    h = fnv(h, directory, n * 64); buffer = alloc(c, 4 * 1024 * 1024); seekto(c, c->spool, 0); remaining = c->end;
    while (remaining) {
        size_t k = remaining > 4 * 1024 * 1024 ? 4 * 1024 * 1024 : (size_t)remaining;
        if (fread(buffer, 1, k, c->spool) != k) die(c, "spool read failed");
        writebytes(c, c->output, buffer, k); h = fnv(h, buffer, k); remaining -= k;
    }
    put(header, h, 8); writebytes(c, c->output, header, 8);
    if (fflush(c->output)) die(c, "output flush failed");
    { int rc = fclose(c->output); c->output = NULL; if (rc) die(c, "output close failed"); }
    fprintf(stderr, "[asr] %llu bytes, %llu entries, %llu fully verified shared tensors, fnv=%016llx\n", (unsigned long long)(filelen + 8), (unsigned long long)n, (unsigned long long)c->shared, (unsigned long long)h);
}
static void cleanup(Ctx *c) {
    if (c->input) fclose(c->input);
    if (c->output) fclose(c->output);
    if (c->spool) fclose(c->spool);
    while (c->mem) { Mem *m = c->mem; c->mem = m->next; free(m); }
    free(c);
}
int vqf_asr_is_bundle(const char *dir) {
    Ctx *c = calloc(1, sizeof(*c)); volatile int yes = 0;
    if (!c) return 0;
    if (!setjmp(c->fail)) {
        char *path = pathjoin(c, dir, "metadata.json"); FILE *f = openfile(c, path, 0);
        if (f) {
            char *json; Slice value; fclose(f); json = jsonfile(c, dir, "metadata.json"); value = jget(c, strslice(json), "format");
            yes = equal(value, "\"audio8_asr_onnx_bundle\"");
        }
    }
    cleanup(c); return yes;
}
int vqf_asr_convert(const char *dir, const char *out, int w8) {
    Ctx *c = calloc(1, sizeof(*c)); volatile int rc = 1;
    if (!c) return 1;
    if (!setjmp(c->fail)) {
        const char *labels[] = {"audio_hidden", "lm_cache_prefill", "lm_cache_decode"};
        char *meta = jsonfile(c, dir, "metadata.json"), *graphs[3], *heads[3], *embedding, *proj, *tokenizer, *feature;
        Slice m = strslice(meta), weights; Str result = {0}; unsigned i;
        if (!equal(jget(c, m, "format"), "\"audio8_asr_onnx_bundle\"")) die(c, "wrong bundle format");
        c->spool = tmpfile(); if (!c->spool) die(c, "cannot create temporary payload spool");
        for (i = 0; i < 3; i++) {
            Slice info = jget(c, jget(c, m, "graphs"), labels[i]);
            graphs[i] = graph(c, dir, jpath(c, jget(c, info, "path")), &heads[i]);
        }
        weights = jget(c, m, "weights");
        { Slice raw = readfile(c, pathjoin(c, dir, jpath(c, jget(c, weights, "token_embedding")))); embedding = npy(c, raw); release(c, (void *)raw.p); }
        for (i = 1; i < 3; i++) if (!heads[i] || strcmp(heads[i], embedding)) die(c, "lm_head/embedding sharing verification failed");
        proj = projector(c, pathjoin(c, dir, jpath(c, jget(c, weights, "audio_projector"))));
        tokenizer = jsonfile(c, dir, "tokenizer.json"); feature = jsonfile(c, dir, "qwen3_asr_feature_extractor/preprocessor_config.json");
        text(c, &result, "{\"format\":\"experimental-asr-v6\",\"schema\":1,\"bundle\":"); text(c, &result, meta); text(c, &result, ",\"graphs\":{");
        for (i = 0; i < 3; i++) { if (i) text(c, &result, ","); quoted(c, &result, strslice(labels[i])); text(c, &result, ":"); text(c, &result, graphs[i]); }
        text(c, &result, "},\"assets\":{\"tokenizer\":"); text(c, &result, tokenizer); text(c, &result, ",\"feature\":"); text(c, &result, feature);
        text(c, &result, "},\"embedding\":"); quoted(c, &result, strslice(embedding)); text(c, &result, ",\"projector\":{"); text(c, &result, proj); text(c, &result, "}");
        if (w8) quantize(c);
        finish(c, &result, out); rc = 0;
    } else fprintf(stderr, "[asr] conversion failed: %s\n", c->error);
    cleanup(c); return rc;
}

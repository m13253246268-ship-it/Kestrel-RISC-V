/* ptrace PC 采样器（板端无 perf/gdb 时用）：pcsamp <pid> <samples> <interval_us>
 * 每次采样中断所有线程读 pc，输出「lib 偏移 计数」，配合 nm 解析符号。 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <elf.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/uio.h>

#define MAXT 64
#define MAXH 65536
static unsigned long hpc[MAXH]; static int hcnt[MAXH]; static int nh;

static void add(unsigned long pc) {
    for (int i = 0; i < nh; i++) if (hpc[i] == pc) { hcnt[i]++; return; }
    if (nh < MAXH) { hpc[nh] = pc; hcnt[nh++] = 1; }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: pcsamp pid samples interval_us\n"); return 1; }
    int pid = atoi(argv[1]), ns = atoi(argv[2]), iv = atoi(argv[3]);
    int tids[MAXT], nt = 0; char p[64];
    snprintf(p, sizeof p, "/proc/%d/task", pid);
    DIR *d = opendir(p); struct dirent *e;
    if (!d) { perror("opendir"); return 1; }
    while ((e = readdir(d)) && nt < MAXT) if (e->d_name[0] != '.') {
        int t = atoi(e->d_name);
        if (ptrace(PTRACE_SEIZE, t, 0, 0) == 0) tids[nt++] = t;
    }
    closedir(d);
    for (int s = 0; s < ns; s++) {
        usleep(iv);
        for (int i = 0; i < nt; i++) {
            int st;
            if (ptrace(PTRACE_INTERRUPT, tids[i], 0, 0)) continue;
            if (waitpid(tids[i], &st, __WALL) < 0) continue;
            unsigned long r[32]; struct iovec io = { r, sizeof r };   /* riscv: r[0]=pc */
            if (ptrace(PTRACE_GETREGSET, tids[i], (void *)NT_PRSTATUS, &io) == 0) add(r[0]);
            ptrace(PTRACE_CONT, tids[i], 0, 0);
        }
    }
    for (int i = 0; i < nt; i++) ptrace(PTRACE_DETACH, tids[i], 0, 0);
    /* 映射到模块偏移 */
    snprintf(p, sizeof p, "/proc/%d/maps", pid);
    FILE *f = fopen(p, "r"); char line[512];
    unsigned long lo[256], hi[256], off[256]; char nm[256][160]; int nm_n = 0;
    while (f && fgets(line, sizeof line, f) && nm_n < 256) {
        char perm[8], path[160] = ""; unsigned long a, b, o;
        if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %159s", &a, &b, perm, &o, path) >= 4 && perm[2] == 'x') {
            lo[nm_n] = a; hi[nm_n] = b; off[nm_n] = o; strcpy(nm[nm_n], path); nm_n++;
        }
    }
    if (f) fclose(f);
    printf("threads=%d samples=%d\n", nt, ns);
    for (int i = 0; i < nh; i++) {
        const char *m = "?"; unsigned long o = hpc[i];
        for (int j = 0; j < nm_n; j++) if (hpc[i] >= lo[j] && hpc[i] < hi[j]) { m = nm[j]; o = hpc[i] - lo[j] + off[j]; break; }
        printf("%d %s 0x%lx\n", hcnt[i], m, o);
    }
    return 0;
}

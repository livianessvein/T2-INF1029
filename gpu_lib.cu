/**
 * @file gpu_lib.cu
 * @brief Biblioteca de eliminação gaussiana com suporte a multithread (host) e GPU (CUDA).
 *
 * Implementa duas versões da funcao processaVetores:
 * - processaVetoresThread: eliminação gaussiana multithread no host usando pthreads.
 * - processaVetoresGPU: eliminação gaussiana acelerada na GPU usando CUDA.
 *
 * A operação realizada eh a triangularização superior da matriz A aumentada [A|b],
 * preservando o sistema de equações lineares para posterior resolução por
 * substituição regressiva.
 *
 * @author  (Grupo)
 * @date    2026
 *
 * @section Códigos de Erro
 * -  1: Falha ao alocar memória no host (hmA ou hvB)
 * -  2: Falha ao alocar memória no device (cuda malloc)
 * -  3: Falha ao copiar dados do host para o device (cudaMemcpy H2D)
 * -  4: Falha ao copiar dados do device para o host (cudaMemcpy D2H)
 * -  5: Falha ao criar thread POSIX
 * -  6: Falha ao fazer join de thread POSIX
 * -  7: Erro de execução do kernel CUDA (verificado via cudaGetLastError)
 * -  8: Falha ao liberar memória no device (cudaFree)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include "comum.h"
#include "gpu.h"

/* =========================================================================
 * SEÇÃO 1 – Versao Multithread no Host (pthreads)
 * =========================================================================
 *
 * Estrategia:
 *   Para cada passo k (linha pivo k-1), as linhas k..n-1 precisam ser
 *   atualizadas. Dividimos essas linhas igualmente entre as nThreads threads.
 *   Cada thread recebe um subconjunto de linhas, calcula o multiplicador e
 *   subtrai a linha pivo multiplicada.
 *
 *   A sincronização é feita no host: o loop principal (passo) avança apenas
 *   após todas as threads terminarem o passo anterior (pthread_join).
 *   Isso evita qualquer condição de corrida sem necessidade de mutex.
 */

/**
 * @brief Funcao executada por cada thread worker na versão host.
 *
 * Cada thread processa um subconjunto de linhas no passo atual da eliminação
 * gaussiana. A linha pivo é passo-1; as linhas abaixo dela são divididas
 * ciclicamente entre as threads (thread i processa linhas i, i+nThreads, …).
 *
 * @param arg  Ponteiro para threadArgs_t com os parâmetros da thread.
 * @return     NULL (sem retorno útil).
 */
static void *gaussWorker(void *arg) {
    threadArgs_t *a = (threadArgs_t *)arg;

    int id        = a->threadId;
    int n         = a->nIncognitas;
    int passo     = a->passo;

    extern int nThreads;
    int nT        = nThreads;

    data_t *mA    = a->hmA;
    data_t *vB    = a->hvB;

    int pivo = passo - 1;

    /* Cada thread trata linhas: passo + id, passo + id + nT, ... */
    for (int linha = passo + id; linha < n; linha += nT) {
        data_t mult = matriz(mA, linha, pivô, n) / matriz(mA, pivô, pivô, n);
        for (int col = pivô; col < n; col++) {
            matriz(mA, linha, col, n) -= matriz(mA, pivô, col, n) * mult;
        }
        vB[linha] -= vB[pivô] * mult;
    }

    return NULL;
}

/**
 * @brief Eliminação gaussiana multithread no host.
 *
 * Realiza a triangularização superior de [A|b] usando nThreads threads POSIX.
 * Em cada passo k, as threads atualizam em paralelo as linhas k..n-1.
 * A sincronização entre passos é garantida por pthread_join.
 *
 * @param hmA         Ponteiro para a matriz A (n×n, row-major, tipo data_t).
 * @param hvB         Ponteiro para o vetor b (tamanho n, tipo data_t).
 * @param nIncognitas Dimensão do sistema (n).
 *
 * @note A variável global ::nThreads controla o número de threads criadas.
 *       Ela é configurada pelo argumento -T da linha de comando.
 *
 * Códigos de erro usados:
 *   5 – falha no pthread_create
 *   6 – falha no pthread_join
 */
void processaVetoresThread(data_t *hmA, data_t *hvB, int nIncognitas) {
    /* Usa a variável global nThreads (definida em equation_test.cu) */
    extern int nThreads;
    int nT = nThreads;

    pthread_t    *threads = (pthread_t *)   malloc(nT * sizeof(pthread_t));
    threadArgs_t *args    = (threadArgs_t *)malloc(nT * sizeof(threadArgs_t));

    if (!threads || !args) {
        fprintf(stderr, "[ERROR] processaVetoresThread: falha ao alocar estruturas de threads\n");
        exit(1);
    }

    for (int passo = 1; passo < nIncognitas; passo++) {
        /* Monta os argumentos e lança as threads */
        for (int t = 0; t < nT; t++) {
            args[t].threadId    = t;
            args[t].hmA         = hmA;
            args[t].hvB         = hvB;
            args[t].nIncognitas = nIncognitas;
            args[t].passo       = passo;

            if (pthread_create(&threads[t], NULL, gaussWorker, &args[t]) != 0) {
                fprintf(stderr, "[ERROR 5] processaVetoresThread: falha em pthread_create (thread %d, passo %d)\n", t, passo);
                exit(5);
            }
        }

        /* Aguarda todas as threads terminarem antes do próximo passo */
        for (int t = 0; t < nT; t++) {
            if (pthread_join(threads[t], NULL) != 0) {
                fprintf(stderr, "[ERROR 6] processaVetoresThread: falha em pthread_join (thread %d, passo %d)\n", t, passo);
                exit(6);
            }
        }
    }

    free(threads);
    free(args);
}

/* =========================================================================
 * SEÇÃO 2 – Versão GPU (CUDA)
 * =========================================================================
 *
 * Estratégia (Gaussian Elimination em paralelo na GPU):
 *
 *   Para cada passo k (linha pivô = k-1):
 *     1. Um kernel calcula o multiplicador de cada linha i >= k e realiza a
 *        subtração linha a linha em paralelo.
 *
 *   O kernel gaussStepKernel recebe o passo atual e atualiza todas as
 *   linhas abaixo do pivô simultaneamente.
 *
 *   Mapeamento de threads:
 *     - Eixo X do bloco → coluna da matriz.
 *     - Eixo Y do grid  → linha da matriz abaixo do pivô.
 *   Dessa forma cada thread (bx, gy) atualiza o elemento (linha, coluna)
 *   de forma independente, sem necessidade de sincronização intra-kernel.
 *
 *   O loop de passos é serializado no host (cada passo lança um kernel);
 *   dentro de cada passo todas as atualizações são paralelas.
 */

/**
 * @brief Kernel CUDA: executa um passo da eliminação gaussiana.
 *
 * Cada thread é responsável por um elemento (linha, coluna) da matriz.
 * Threads cujo índice de linha é <= passo-1 não fazem nada.
 *
 * @param dmA         Matriz A no device (n×n, row-major).
 * @param dvB         Vetor b no device (tamanho n).
 * @param n           Dimensão do sistema.
 * @param passo       Passo atual (linha pivô = passo-1).
 */
__global__ void gaussStepKernel(data_t *dmA, data_t *dvB, int n, int passo) {
    /* Índice global de linha: linhas abaixo do pivô */
    int linha  = blockIdx.y * blockDim.y + threadIdx.y + passo;
    /* Índice global de coluna */
    int coluna = blockIdx.x * blockDim.x + threadIdx.x;

    if (linha >= n || coluna >= n) return;

    int pivô = passo - 1;

    /* Calcula o multiplicador (cada thread da mesma linha lê o mesmo valor) */
    data_t mult = dmA[linha * n + pivô] / dmA[pivô * n + pivô];

    /* Atualiza o elemento da matriz */
    dmA[linha * n + coluna] -= dmA[pivô * n + coluna] * mult;

    /* Somente a thread da coluna 0 atualiza o vetor b */
    if (coluna == 0) {
        dvB[linha] -= dvB[pivô] * mult;
    }
}

/**
 * @brief Macro auxiliar para verificar erros de CUDA e encerrar o programa.
 *
 * @param call  Chamada à API CUDA a ser verificada.
 * @param code  Código de erro a ser passado para exit() em caso de falha.
 */
#define CUDA_CHECK(call, code)                                                       \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "[ERROR %d] CUDA: %s (%s:%d)\n",                        \
                    (code), cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(code);                                                              \
        }                                                                            \
    } while (0)

/**
 * @brief Eliminação gaussiana acelerada na GPU (CUDA).
 *
 * Copia a matriz A e o vetor b para o device, executa a eliminação gaussiana
 * em paralelo na GPU e copia os resultados de volta para o host.
 *
 * Configuração de lançamento:
 *   - threadsPerBlock (global, argumento -g): threads no eixo X (colunas).
 *   - blocksPerGrid   (global, argumento -t): máximo de blocos no eixo X.
 *   - No eixo Y, blocos e threads são ajustados para cobrir as linhas abaixo
 *     do pivô a cada passo.
 *
 * @param hmA         Ponteiro para a matriz A no host (n×n, row-major).
 * @param hvB         Ponteiro para o vetor b no host (tamanho n).
 * @param nIncognitas Dimensão do sistema (n).
 *
 * Códigos de erro usados:
 *   2 – falha no cudaMalloc
 *   3 – falha no cudaMemcpy (H2D)
 *   4 – falha no cudaMemcpy (D2H)
 *   7 – erro de execução de kernel
 *   8 – falha no cudaFree
 */
void processaVetoresGPU(data_t *hmA, data_t *hvB, int nIncognitas) {
    extern int threadsPerBlock;
    extern int blocksPerGrid;

    int n = nIncognitas;

    /* ---- Aloca memória no device ---- */
    data_t *dmA = NULL;
    data_t *dvB = NULL;

    CUDA_CHECK(cudaMalloc((void **)&dmA, sizeof(data_t) * n * n), 2);
    CUDA_CHECK(cudaMalloc((void **)&dvB, sizeof(data_t) * n),     2);

    /* ---- Copia dados do host para o device ---- */
    CUDA_CHECK(cudaMemcpy(dmA, hmA, sizeof(data_t) * n * n, cudaMemcpyHostToDevice), 3);
    CUDA_CHECK(cudaMemcpy(dvB, hvB, sizeof(data_t) * n,     cudaMemcpyHostToDevice), 3);

    /* ---- Loop de passos (serializado no host) ---- */
    /*
     * Configuração 2D:
     *   Eixo X: colunas da matriz   → usamos threadsPerBlock threads por bloco
     *   Eixo Y: linhas abaixo pivô  → 1 thread por linha (blocos de 1 linha)
     *
     * Limitamos o número de blocos em X ao mínimo necessário para cobrir n colunas
     * e o número de blocos em Y ao número de linhas a processar neste passo.
     */
    int tpbX = menor(threadsPerBlock, n);           /* threads por bloco em X */
    int tpbY = 1;                                   /* threads por bloco em Y */

    for (int passo = 1; passo < n; passo++) {
        int linhasAtivas = n - passo;               /* linhas abaixo do pivô  */

        int bX = (n          + tpbX - 1) / tpbX;   /* blocos necessários em X */
        int bY = linhasAtivas;                      /* um bloco por linha      */

        /* Aplica o limite global de blocos por grid em X */
        bX = menor(bX, blocksPerGrid);

        dim3 blockDim(tpbX, tpbY);
        dim3 gridDim(bX, bY);

        gaussStepKernel<<<gridDim, blockDim>>>(dmA, dvB, n, passo);

        /* Verifica erros de execução do kernel */
        CUDA_CHECK(cudaGetLastError(), 7);
    }

    /* Aguarda término de todos os kernels antes de copiar o resultado */
    CUDA_CHECK(cudaDeviceSynchronize(), 7);

    /* ---- Copia resultados de volta para o host ---- */
    CUDA_CHECK(cudaMemcpy(hmA, dmA, sizeof(data_t) * n * n, cudaMemcpyDeviceToHost), 4);
    CUDA_CHECK(cudaMemcpy(hvB, dvB, sizeof(data_t) * n,     cudaMemcpyDeviceToHost), 4);

    /* ---- Libera memória no device ---- */
    CUDA_CHECK(cudaFree(dmA), 8);
    CUDA_CHECK(cudaFree(dvB), 8);
}

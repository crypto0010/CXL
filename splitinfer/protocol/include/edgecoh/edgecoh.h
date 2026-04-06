/* splitinfer/protocol/include/edgecoh/edgecoh.h */
#ifndef EDGECOH_H
#define EDGECOH_H

#include "messages.h"
#include <stdint.h>

#define EDGECOH_MAX_TENSORS 256

typedef struct {
    uint8_t  owner[EDGECOH_MAX_TENSORS];
    uint8_t  prefetch_pending[EDGECOH_MAX_TENSORS];
    uint8_t  prefetch_target[EDGECOH_MAX_TENSORS];
    uint16_t num_tensors;
    uint32_t epoch;
} edgecoh_state_t;

void edgecoh_init(edgecoh_state_t *state, uint16_t num_tensors);
int edgecoh_transfer(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device);
int edgecoh_get_owner(const edgecoh_state_t *state, uint16_t tensor_id);
int edgecoh_prefetch(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device);
int edgecoh_prefetch_complete(edgecoh_state_t *state, uint16_t tensor_id);
int edgecoh_is_prefetching(const edgecoh_state_t *state, uint16_t tensor_id);
void edgecoh_barrier(edgecoh_state_t *state);
uint32_t edgecoh_get_epoch(const edgecoh_state_t *state);

#endif /* EDGECOH_H */

/* splitinfer/protocol/src/edgecoh.c */
#include "edgecoh/edgecoh.h"
#include <string.h>

void edgecoh_init(edgecoh_state_t *state, uint16_t num_tensors) {
    memset(state, 0, sizeof(*state));
    state->num_tensors = (num_tensors > EDGECOH_MAX_TENSORS)
                         ? EDGECOH_MAX_TENSORS : num_tensors;
    state->epoch = 0;
}

int edgecoh_transfer(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device) {
    if (tensor_id >= state->num_tensors) return -1;
    state->owner[tensor_id] = target_device;
    return 0;
}

int edgecoh_get_owner(const edgecoh_state_t *state, uint16_t tensor_id) {
    if (tensor_id >= state->num_tensors) return -1;
    return state->owner[tensor_id];
}

int edgecoh_prefetch(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device) {
    if (tensor_id >= state->num_tensors) return -1;
    state->prefetch_pending[tensor_id] = 1;
    state->prefetch_target[tensor_id] = target_device;
    return 0;
}

int edgecoh_prefetch_complete(edgecoh_state_t *state, uint16_t tensor_id) {
    if (tensor_id >= state->num_tensors) return -1;
    if (!state->prefetch_pending[tensor_id]) return -1;
    state->owner[tensor_id] = state->prefetch_target[tensor_id];
    state->prefetch_pending[tensor_id] = 0;
    return 0;
}

int edgecoh_is_prefetching(const edgecoh_state_t *state, uint16_t tensor_id) {
    if (tensor_id >= state->num_tensors) return -1;
    return state->prefetch_pending[tensor_id];
}

void edgecoh_barrier(edgecoh_state_t *state) {
    state->epoch++;
}

uint32_t edgecoh_get_epoch(const edgecoh_state_t *state) {
    return state->epoch;
}

/* splitinfer/protocol/tests/test_state_machine.c */
#include <stdio.h>
#include <assert.h>
#include "edgecoh/edgecoh.h"

static void test_initial_state(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    assert(edgecoh_get_owner(&state, 0) == EDGECOH_DEV_HOST);
    assert(edgecoh_get_owner(&state, 15) == EDGECOH_DEV_HOST);
}

static void test_transfer_ownership(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    int rc = edgecoh_transfer(&state, 5, EDGECOH_DEV_FPGA);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 5) == EDGECOH_DEV_FPGA);
    rc = edgecoh_transfer(&state, 5, EDGECOH_DEV_HOST);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 5) == EDGECOH_DEV_HOST);
}

static void test_transfer_to_same_owner_is_noop(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    int rc = edgecoh_transfer(&state, 3, EDGECOH_DEV_HOST);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 3) == EDGECOH_DEV_HOST);
}

static void test_invalid_tensor_id(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    int rc = edgecoh_transfer(&state, 99, EDGECOH_DEV_FPGA);
    assert(rc == -1);
    assert(edgecoh_get_owner(&state, 99) == -1);
}

static void test_prefetch_marks_pending(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    int rc = edgecoh_prefetch(&state, 8, EDGECOH_DEV_FPGA);
    assert(rc == 0);
    assert(edgecoh_is_prefetching(&state, 8) == 1);
    rc = edgecoh_prefetch_complete(&state, 8);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 8) == EDGECOH_DEV_FPGA);
    assert(edgecoh_is_prefetching(&state, 8) == 0);
}

static void test_barrier_increments_epoch(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    uint32_t epoch0 = edgecoh_get_epoch(&state);
    edgecoh_barrier(&state);
    uint32_t epoch1 = edgecoh_get_epoch(&state);
    assert(epoch1 == epoch0 + 1);
}

int main(void) {
    test_initial_state();
    test_transfer_ownership();
    test_transfer_to_same_owner_is_noop();
    test_invalid_tensor_id();
    test_prefetch_marks_pending();
    test_barrier_increments_epoch();
    printf("All state machine tests passed.\n");
    return 0;
}

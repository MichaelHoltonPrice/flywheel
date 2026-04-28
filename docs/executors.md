# Executor protocol

`flywheel.executor` contains the shared protocol/result surface:

* `BlockExecutor`
* `ExecutionHandle`
* `SyncExecutionHandle`
* `ExecutionResult`
* `ExecutionEvent`

The canonical ad hoc block-execution surface is
`flywheel.execution.run_block`. It owns input resolution, proposal
allocation, container invocation, artifact forging, quarantine, and
ledger recording for supported container lifecycles.

Concrete container execution is not implemented in `flywheel.executor`.

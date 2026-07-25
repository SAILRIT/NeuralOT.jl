# Shared training-loop plumbing: logging, progress, callbacks, early stopping.

"""
    _log_step(step, log_every, steps) -> Bool

Whether this step should be logged: the first, the last, and every
`log_every`-th in between.
"""
_log_step(step::Int, log_every::Int, steps::Int) =
    log_every > 0 && (step == 1 || step == steps || step % log_every == 0)

"""
    _report(verbose, method, step, steps, loss, evalloss)

Print a one-line progress report when `verbose` is true.
"""
function _report(verbose::Bool, method::Symbol, step::Int, steps::Int,
                 loss::Real, evalloss)
    verbose || return nothing
    if evalloss === nothing
        @printf("[%s] step %6d/%d  loss %12.6g\n", method, step, steps, loss)
    else
        @printf("[%s] step %6d/%d  loss %12.6g  eval %12.6g\n", method, step, steps,
                loss, evalloss)
    end
    flush(stdout)
    return nothing
end

"""
    _run_callback(callback, step, loss, models) -> Bool

Invoke a user callback, returning `true` to continue training. A callback that
returns `false` (and only `false`) stops training early; any other return value
is ignored.
"""
function _run_callback(callback, step::Int, loss::Real, models)
    callback === nothing && return true
    out = callback(step, loss, models)
    return out === false ? false : true
end

"""
    _validate_common(steps, batch, lr, log_every)

Argument checking shared by every solver, so that a typo fails immediately
rather than after a long training run.
"""
function _validate_common(steps::Int, batch::Int, lr::Real, log_every::Int)
    steps >= 1 || throw(ArgumentError("steps must be at least 1, got $steps"))
    batch >= 2 || throw(ArgumentError("batch must be at least 2, got $batch"))
    lr > 0 || throw(ArgumentError("lr must be positive, got $lr"))
    log_every >= 0 || throw(ArgumentError("log_every must be non-negative, got $log_every"))
    return nothing
end

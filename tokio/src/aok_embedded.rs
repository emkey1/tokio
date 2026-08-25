//! iSH-AOK embedded mode (feature `aok-embedded`).
//!
//! In iSH-AOK a "native program" is not a process: it is a function called on
//! a guest task's thread inside the app, possibly many times, possibly from
//! several tasks at once. Anything tokio keeps `static` therefore outlives the
//! program run that created it — and tokio's signal support keeps exactly such
//! state: the wakeup socketpair in `signal::registry::globals()` and the
//! orphan-process queue in `process::unix`. Both hold file descriptors, and in
//! iSH-AOK an fd is scoped to the guest TASK that opened it. The second run of
//! a tokio program found the first run's dead socketpair and
//! `Runtime::build()` died with EBADF before user code ran.
//!
//! The fix is to key those statics by *invocation*: iSH-AOK exports
//! `nlibc_invocation_token()` (kernel/native.h), a value distinct for every
//! run of every native program, shared by all threads a run creates, and
//! async-signal-safe to read. `PerInvocation<T>` below maps that token to a
//! leaked `T`.
//!
//! Values are leaked, never dropped, on purpose. Dropping a previous run's
//! value would run `close()` on its cached fd numbers *through the current
//! task's descriptor table*, closing whatever unrelated files happen to have
//! those numbers now. The previous run's real resources were already reclaimed
//! when its task exited; the leak is a few hundred bytes per program run.
//!
//! Lookup is a lock-free walk of an append-only list, so it is safe from a
//! signal handler. Creation takes a mutex, but a handler can never reach it:
//! a handler only runs after `signal_enable()` registered it, and that call
//! already created this run's entry on the same thread's fast path.

use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

extern "C" {
    fn nlibc_invocation_token() -> u64;
}

/// The current native-program run's token. 0 means the calling thread does not
/// belong to a native program, which for tokio-in-AOK should not happen; it
/// would merely make such threads share one extra state generation.
pub(crate) fn invocation_token() -> u64 {
    // SAFETY: reads one __thread variable; provided by iSH-AOK's libc shim.
    unsafe { nlibc_invocation_token() }
}

struct Node<T> {
    token: u64,
    value: T,
    next: *const Node<T>,
}

/// An append-only map from invocation token to a leaked `T`.
pub(crate) struct PerInvocation<T> {
    head: AtomicPtr<Node<T>>,
    grow: Mutex<()>,
}

// SAFETY: values are only shared, never handed out mutably; nodes are never
// freed.
unsafe impl<T: Send + Sync> Send for PerInvocation<T> {}
unsafe impl<T: Send + Sync> Sync for PerInvocation<T> {}

impl<T> PerInvocation<T> {
    pub(crate) const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            grow: Mutex::new(()),
        }
    }

    fn find(&self, token: u64) -> Option<&'static T> {
        let mut p = self.head.load(Ordering::Acquire) as *const Node<T>;
        while !p.is_null() {
            // SAFETY: nodes are leaked; a pointer read from the list is valid
            // forever.
            let node = unsafe { &*p };
            if node.token == token {
                return Some(&node.value);
            }
            p = node.next;
        }
        None
    }

    /// This invocation's value, created with `init` on first use.
    pub(crate) fn get_or_init(&self, init: impl FnOnce() -> T) -> &'static T {
        let token = invocation_token();
        if let Some(value) = self.find(token) {
            return value;
        }
        let guard = self.grow.lock().unwrap();
        // Another thread of this run may have created it while we waited.
        if let Some(value) = self.find(token) {
            return value;
        }
        let node = Box::leak(Box::new(Node {
            token,
            value: init(),
            next: self.head.load(Ordering::Relaxed),
        }));
        self.head.store(node, Ordering::Release);
        drop(guard);
        &node.value
    }
}

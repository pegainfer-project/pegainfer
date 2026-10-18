use std::cell::Cell;
use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::OnceLock;

use anyhow::Result;
use pegainfer_kernels::tensor::KernelCall;

thread_local! {
    static TRACE: RefCell<Option<Vec<KernelCall>>> = const { RefCell::new(None) };
    static LABEL_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static DECODE_KV_LEN: Cell<Option<usize>> = const { Cell::new(None) };
}

static GLOBAL_TRACE: OnceLock<Mutex<Option<Vec<KernelCall>>>> = OnceLock::new();

fn global_trace() -> &'static Mutex<Option<Vec<KernelCall>>> {
    GLOBAL_TRACE.get_or_init(|| Mutex::new(None))
}

pub fn collect_result<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Vec<KernelCall>)> {
    TRACE.with(|trace| {
        let previous = trace.replace(Some(Vec::new()));
        assert!(
            previous.is_none(),
            "nested kernel call trace collection is not supported"
        );
    });
    {
        let mut global = global_trace()
            .lock()
            .expect("kernel call global trace mutex poisoned");
        assert!(
            global.is_none(),
            "nested global kernel call trace collection is not supported"
        );
        *global = Some(Vec::new());
    }

    let result = f();
    let calls = TRACE.with(|trace| trace.replace(None).unwrap_or_default());
    let global_calls = global_trace()
        .lock()
        .expect("kernel call global trace mutex poisoned")
        .take()
        .unwrap_or_default();
    result.map(|value| {
        let mut all_calls = calls;
        all_calls.extend(global_calls);
        (value, all_calls)
    })
}

pub fn is_enabled() -> bool {
    TRACE.with(|trace| trace.borrow().is_some())
        || global_trace()
            .lock()
            .expect("kernel call global trace mutex poisoned")
            .is_some()
}

pub fn record_call(call: KernelCall) {
    let recorded = TRACE.with(|trace| {
        if let Some(calls) = trace.borrow_mut().as_mut() {
            calls.push(call.clone());
            true
        } else {
            false
        }
    });
    if recorded {
        return;
    }
    if let Some(calls) = global_trace()
        .lock()
        .expect("kernel call global trace mutex poisoned")
        .as_mut()
    {
        calls.push(call);
    }
}

pub fn current_label(default_op: &str) -> String {
    LABEL_STACK.with(|stack| {
        stack
            .borrow()
            .last()
            .cloned()
            .unwrap_or_else(|| default_op.to_string())
    })
}

pub fn with_decode_kv_len<T>(kv_len: usize, f: impl FnOnce() -> T) -> T {
    DECODE_KV_LEN.with(|cell| {
        let previous = cell.replace(Some(kv_len));
        let result = f();
        cell.set(previous);
        result
    })
}

pub fn decode_kv_len() -> Option<usize> {
    DECODE_KV_LEN.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_result_captures_calls_from_child_thread() {
        let ((), calls) = collect_result(|| {
            std::thread::spawn(|| {
                record_call(KernelCall::new("child_op", "child.label"));
            })
            .join()
            .expect("child thread");
            Ok(())
        })
        .expect("collect trace");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].op, "child_op");
        assert_eq!(calls[0].label, "child.label");
    }
}

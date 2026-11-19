//! Pins what a torn record must not do: parse as a hit, or take the record
//! written after it down with itself.
//!
//! `RLIMIT_FSIZE` produces a real short write rather than an imitation of one.
//! `SIGXFSZ` has to be ignored first or the kernel kills the process instead of
//! letting the write return.
use std::ffi::CString;
use std::fs;
use std::sync::Mutex;

use pegainfer_kernels::ffi::pegainfer_lt_store_append;
use pegainfer_kernels::ffi::pegainfer_lt_store_lookup;

fn lookup(path: &CString, key: &str) -> Option<[u64; 8]> {
    let k = CString::new(key).unwrap();
    let mut out = [0u64; 8];
    let found = unsafe { pegainfer_lt_store_lookup(path.as_ptr(), k.as_ptr(), out.as_mut_ptr()) };
    (found == 1).then_some(out)
}

fn append(path: &CString, key: &str, words: &[u64; 8]) {
    let k = CString::new(key).unwrap();
    unsafe { pegainfer_lt_store_append(path.as_ptr(), k.as_ptr(), words.as_ptr()) };
}

// RLIMIT_FSIZE and the SIGXFSZ disposition are process-wide, and cargo runs a
// binary's tests in parallel, so the window where the limit is low would truncate
// whatever the other test happens to be writing.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn set_file_size_limit(bytes: u64) {
    let lim = libc::rlimit {
        rlim_cur: bytes,
        rlim_max: libc::RLIM_INFINITY,
    };
    assert_eq!(
        unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &raw const lim) },
        0,
        "setrlimit"
    );
}

#[test]
fn a_short_write_neither_parses_nor_poisons_what_follows() {
    let _guard = ONE_AT_A_TIME.lock().unwrap();
    unsafe { libc::signal(libc::SIGXFSZ, libc::SIG_IGN) };
    let dir = std::env::temp_dir().join(format!("pegainfer-lt-store-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("store.txt");
    let path = CString::new(file.to_str().unwrap()).unwrap();

    let first = [0x0123_4567_89ab_cdef_u64; 8];
    let second = [0x1122_3344_5566_7788_u64; 8];
    let third = [0xdead_beef_cafe_0001_u64; 8];

    append(&path, "KEY-A", &first);
    assert_eq!(
        lookup(&path, "KEY-A"),
        Some(first),
        "a whole record reads back"
    );

    // Cap the file so the next append is cut mid-record. Nine bytes short is
    // enough to lose part of the last field without touching the key.
    let whole = fs::metadata(&file).unwrap().len();
    let record_len = whole; // both records are the same length
    set_file_size_limit(whole + record_len - 9);
    append(&path, "KEY-B", &second);
    set_file_size_limit(libc::RLIM_INFINITY);

    let torn = fs::read_to_string(&file).unwrap();
    assert!(
        !torn.ends_with('\n'),
        "the second record really was cut short"
    );
    assert_eq!(
        lookup(&path, "KEY-B"),
        None,
        "a record whose last field is truncated must not parse as a hit"
    );
    assert_eq!(
        lookup(&path, "KEY-A"),
        Some(first),
        "the intact record survives"
    );

    // The record after a torn tail has to be readable, which means the append
    // must start a new line rather than continue the broken one.
    append(&path, "KEY-C", &third);
    assert_eq!(
        lookup(&path, "KEY-C"),
        Some(third),
        "the store recovers for later writes"
    );
    assert_eq!(
        lookup(&path, "KEY-A"),
        Some(first),
        "and still holds what it had"
    );
    assert_eq!(
        lookup(&path, "KEY-B"),
        None,
        "the torn record stays unreadable"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_later_record_wins_and_a_malformed_one_is_ignored() {
    let _guard = ONE_AT_A_TIME.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("pegainfer-lt-store-b-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("store.txt");
    let path = CString::new(file.to_str().unwrap()).unwrap();

    let old = [1u64; 8];
    let new = [2u64; 8];
    append(&path, "KEY", &old);
    append(&path, "KEY", &new);
    assert_eq!(
        lookup(&path, "KEY"),
        Some(new),
        "a re-tune appends and the later record wins"
    );

    // Field widths are exact: a short field, a long one, and a trailing extra
    // are each a different record than the one they resemble.
    let mut text = fs::read_to_string(&file).unwrap();
    text.push_str("v1 SHORT 0000000000000001 0000000000000002 0000000000000003 0000000000000004 0000000000000005 0000000000000006 0000000000000007 000000000001\n");
    text.push_str("v1 EXTRA 0000000000000001 0000000000000002 0000000000000003 0000000000000004 0000000000000005 0000000000000006 0000000000000007 0000000000000008 9\n");
    text.push_str("v1 NOTHEX 000000000000000g 0000000000000002 0000000000000003 0000000000000004 0000000000000005 0000000000000006 0000000000000007 0000000000000008\n");
    fs::write(&file, &text).unwrap();
    for key in ["SHORT", "EXTRA", "NOTHEX"] {
        assert_eq!(lookup(&path, key), None, "{key} must not parse");
    }
    assert_eq!(
        lookup(&path, "KEY"),
        Some(new),
        "and the good record is untouched"
    );

    let _ = fs::remove_dir_all(&dir);
}

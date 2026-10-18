//! One-time bootstrap rendezvous for multi-process EP fleets. Every rank in
//! the DeepEP world needs the same `ncclUniqueId`; the process hosting rank
//! 0 generates it and serves it over a minimal TCP handshake, and every
//! other process pulls it once at startup before its `SetupComm`. This is
//! the entire cross-process control plane: after the id is handed out the
//! engines never talk to each other again — synchronization is the
//! collective's back-pressure (`docs/models/glm52/free-running-dp.md` §3).
//!
//! Fail-stop by construction: a connect that never succeeds is a launch
//! failure here; a mis-tiled rank deployment hangs in DeepEP's collective
//! `ctx_create` and every process dies on its device timeout. A restarted
//! process re-fetches the SAME id (the serve side is idempotent — the whole
//! fleet restarts together under fate-sharing, so the id never rotates).

use std::io::Read as _;
use std::io::Write as _;
use std::net::TcpListener;
use std::net::TcpStream;
use std::ops::Range;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;

/// Wire version: bump when the handshake shape changes.
const RENDEZVOUS_VERSION: u32 = 1;
/// Rank-0's process may still be loading its weights when peers first
/// connect; retry for a generous window before declaring a launch failure.
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(2);
const CONNECT_TIMEOUT_TOTAL: Duration = Duration::from_secs(600);
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Obtain this process's DeepEP unique id: generated in-process for a
/// single-process fleet; generated and served by the rank-0-hosting process
/// (`ranks.start == 0` with a rendezvous address); fetched from there by
/// every other process.
pub(crate) fn unique_id(
    ep_size: usize,
    ranks: &Range<usize>,
    rendezvous: Option<&str>,
) -> Result<[u8; 128]> {
    match rendezvous {
        None => generate(ep_size),
        Some(addr) if ranks.start == 0 => {
            let id = generate(ep_size)?;
            serve(addr, ep_size, id)?;
            log::info!(
                "GLM5.2 bootstrap rendezvous: serving DeepEP id for ep_size={ep_size} on {addr} \
                 (this process hosts ranks {}..{})",
                ranks.start,
                ranks.end
            );
            Ok(id)
        }
        Some(addr) => fetch(addr, ep_size, ranks),
    }
}

fn generate(ep_size: usize) -> Result<[u8; 128]> {
    pegainfer_kernels::ops::glm52_ep_deepep_unique_id(ep_size)
        .map_err(|err| anyhow::anyhow!("GLM5.2 DeepEP unique id generation: {err}"))
}

/// Serve the id to every connecting peer until the process dies. The
/// listener thread is deliberately detached: the rendezvous is a
/// process-lifetime facility (restarted peers re-fetch after a fleet-wide
/// failure), and process exit is the release mechanism.
fn serve(addr: &str, ep_size: usize, id: [u8; 128]) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .with_context(|| format!("GLM5.2 bootstrap rendezvous bind {addr}"))?;
    std::thread::Builder::new()
        .name("glm52-rendezvous".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        if let Err(err) = answer(stream, ep_size, &id) {
                            log::warn!("GLM5.2 bootstrap rendezvous answer failed: {err:#}");
                        }
                    }
                    Err(err) => {
                        log::warn!("GLM5.2 bootstrap rendezvous accept failed: {err}");
                    }
                }
            }
        })
        .context("GLM5.2 bootstrap rendezvous thread")?;
    Ok(())
}

fn answer(mut stream: TcpStream, ep_size: usize, id: &[u8; 128]) -> Result<()> {
    let peer = stream
        .peer_addr()
        .map_or_else(|_| "<unknown>".into(), |addr| addr.to_string());
    let hello = read_hello(&mut stream)?;
    let (version, peer_ep, rank_start, rank_end) = hello;
    if version != RENDEZVOUS_VERSION || peer_ep as usize != ep_size {
        let message = format!(
            "rendezvous mismatch: peer version={version} ep_size={peer_ep}, \
             expected version={RENDEZVOUS_VERSION} ep_size={ep_size}"
        );
        write_reply(&mut stream, Err(&message))?;
        bail!("GLM5.2 bootstrap {message}");
    }
    log::info!(
        "GLM5.2 bootstrap rendezvous: peer {peer} checked in ranks {rank_start}..{rank_end}"
    );
    write_reply(&mut stream, Ok(id))
}

fn fetch(addr: &str, ep_size: usize, ranks: &Range<usize>) -> Result<[u8; 128]> {
    // Connect is retried (rank 0 may still be starting); the handshake is
    // terminal — a rejection is a configuration error, never transient.
    let started = Instant::now();
    let mut attempt = 0u32;
    let mut stream = loop {
        attempt += 1;
        match TcpStream::connect(addr) {
            Ok(stream) => break stream,
            Err(err) => {
                anyhow::ensure!(
                    started.elapsed() < CONNECT_TIMEOUT_TOTAL,
                    "GLM5.2 bootstrap rendezvous {addr} unreachable after {}s: {err}",
                    CONNECT_TIMEOUT_TOTAL.as_secs()
                );
                log::info!("GLM5.2 bootstrap rendezvous {addr} not ready ({err}); retrying");
                std::thread::sleep(CONNECT_RETRY_INTERVAL);
            }
        }
    };
    let id = handshake(&mut stream, ep_size, ranks)?;
    log::info!(
        "GLM5.2 bootstrap rendezvous: fetched DeepEP id from {addr} \
         (attempt {attempt}, ranks {}..{})",
        ranks.start,
        ranks.end
    );
    Ok(id)
}

fn handshake(stream: &mut TcpStream, ep_size: usize, ranks: &Range<usize>) -> Result<[u8; 128]> {
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))?;
    let hello = [
        RENDEZVOUS_VERSION.to_le_bytes(),
        (ep_size as u32).to_le_bytes(),
        (ranks.start as u32).to_le_bytes(),
        (ranks.end as u32).to_le_bytes(),
    ]
    .concat();
    stream.write_all(&hello)?;
    let mut status = [0u8; 4];
    stream.read_exact(&mut status)?;
    if u32::from_le_bytes(status) != 0 {
        let mut message = String::new();
        stream.read_to_string(&mut message)?;
        bail!("GLM5.2 bootstrap rendezvous rejected: {message}");
    }
    let mut id = [0u8; 128];
    stream.read_exact(&mut id)?;
    Ok(id)
}

fn read_hello(stream: &mut TcpStream) -> Result<(u32, u32, u32, u32)> {
    let mut hello = [0u8; 16];
    stream.read_exact(&mut hello)?;
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().expect("4-byte word"));
    Ok((
        word(&hello[0..4]),
        word(&hello[4..8]),
        word(&hello[8..12]),
        word(&hello[12..16]),
    ))
}

fn write_reply(stream: &mut TcpStream, reply: Result<&[u8; 128], &str>) -> Result<()> {
    match reply {
        Ok(id) => {
            stream.write_all(&0u32.to_le_bytes())?;
            stream.write_all(id)?;
        }
        Err(message) => {
            stream.write_all(&1u32.to_le_bytes())?;
            stream.write_all(message.as_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_then_fetch_round_trips_the_id() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        let id = [7u8; 128];
        let serve_id = id;
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let _ = answer(stream.expect("accept"), 16, &serve_id);
            }
        });

        let fetched = fetch(&addr, 16, &(4..8)).expect("fetch");
        assert_eq!(fetched, [7u8; 128]);
    }

    #[test]
    fn ep_size_mismatch_is_a_rendezvous_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let _ = answer(stream.expect("accept"), 16, &[0u8; 128]);
            }
        });

        let mut stream = TcpStream::connect(&addr).expect("connect");
        let err = handshake(&mut stream, 8, &(4..8)).expect_err("mismatch must fail");
        assert!(err.to_string().contains("rejected"), "{err:#}");
    }
}

use std::path::PathBuf;

use pegaflow_core::EngineError;
use pegainfer_kv_store::P2pConfig;
use pegainfer_kv_store::PegaflowHost;

#[tokio::test]
async fn invalid_host_config_returns_an_error_in_async_context() {
    const PINNED_POOL_BYTES: usize = 4096;
    const SSD_CAPACITY_BYTES: u64 = 4096;
    let cases = [
        (
            PegaflowHost::builder(PINNED_POOL_BYTES).ssd_cache(vec![], SSD_CAPACITY_BYTES),
            "ssd_cache requires at least one cache path",
        ),
        (
            PegaflowHost::builder(PINNED_POOL_BYTES)
                .ssd_cache(vec![PathBuf::from("unused-cache-path")], 0),
            "ssd_cache capacity must be non-zero",
        ),
        (
            PegaflowHost::builder(PINNED_POOL_BYTES).p2p(P2pConfig {
                metaserver_addr: "http://127.0.0.1:50056".into(),
                advertise_addr: "127.0.0.1:0".into(),
                rdma_nics: vec![],
            }),
            "P2P requires at least one RDMA NIC",
        ),
    ];
    for (builder, expected) in cases {
        match builder.build() {
            Err(EngineError::InvalidArgument(message)) => assert_eq!(message, expected),
            Err(error) => panic!("expected invalid configuration, got {error}"),
            Ok(_) => panic!("invalid configuration created a host"),
        }
    }
}

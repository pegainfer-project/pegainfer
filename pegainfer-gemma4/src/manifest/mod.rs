//! [`schema`] turns a config into the tensor contract the checkpoint must
//! satisfy; [`validate`] holds that contract up against the headers it carries.

use safetensors::Dtype;

pub(crate) mod schema;
pub(crate) mod validate;

const EXPECTED_DTYPE: Dtype = Dtype::BF16;

use std::collections::HashMap;
#[allow(unused_imports)]
use std::env;
use std::fs;
use std::process;

/// One table of numbers from a safetensors file.
#[allow(dead_code)]
struct Tensor {
    dtype: String,
    shape: Vec<u64>,
    values: Vec<f32>,
}

/// Reads every tensor in a safetensors file, with all of its numbers.
#[allow(dead_code)]
fn load_tensors(path: &str) -> HashMap<String, Tensor> {
    let file = fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(file[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&file[8..8 + header_len]).unwrap();
    let data = &file[8 + header_len..];

    let mut tensors = HashMap::new();
    for (name, info) in &header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap().to_string();
        let shape: Vec<u64> = info["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dim| dim.as_u64().unwrap())
            .collect();
        let offsets = info["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        let values: Vec<f32> = data[start..end]
            .chunks_exact(4)
            .map(|chunk| {
                let bits = u32::from_le_bytes(chunk.try_into().unwrap());
                match dtype.as_str() {
                    "F32" => f32::from_bits(bits),
                    "I32" => bits as i32 as f32,
                    _ => {
                        eprintln!("unsupported dtype {dtype}");
                        process::exit(1);
                    }
                }
            })
            .collect();
        tensors.insert(
            name.clone(),
            Tensor {
                dtype,
                shape,
                values,
            },
        );
    }
    tensors
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args[1] == "inspect" {
        let tensors = load_tensors(&args[2]);
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        let summaries: Vec<serde_json::Value> = names
            .iter()
            .map(|name| {
                let tensor = &tensors[*name];
                let first = &tensor.values[..tensor.values.len().min(4)];
                serde_json::json!({
                    "name": name,
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "first": first,
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&summaries).unwrap());
    } else {
        eprintln!("Unknown command {}", args[1]);
        process::exit(1);
    }
}

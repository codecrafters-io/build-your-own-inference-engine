#[allow(unused_imports)]
use std::env;
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::io::Read;
#[allow(unused_imports)]
use std::process;

fn main() {
    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");

    // TODO: Uncomment the code below to pass the first stage
    // let args: Vec<String> = env::args().collect();
    // if args[1] == "inspect" {
    //     let mut file = fs::File::open(&args[2]).unwrap();
    //     let mut len_buf = [0u8; 8];
    //     file.read_exact(&mut len_buf).unwrap();
    //     let header_len = u64::from_le_bytes(len_buf) as usize;
    //     let mut header_buf = vec![0u8; header_len];
    //     file.read_exact(&mut header_buf).unwrap();
    //     let header: serde_json::Map<String, serde_json::Value> =
    //         serde_json::from_slice(&header_buf).unwrap();
    //     let mut data = Vec::new();
    //     file.read_to_end(&mut data).unwrap();
    //
    //     let mut names: Vec<&String> = header
    //         .keys()
    //         .filter(|name| name.as_str() != "__metadata__")
    //         .collect();
    //     names.sort();
    //
    //     let mut tensors = Vec::new();
    //     for name in names {
    //         let info = &header[name];
    //         let dtype = info["dtype"].as_str().unwrap();
    //         let shape: Vec<u64> = info["shape"]
    //             .as_array()
    //             .unwrap()
    //             .iter()
    //             .map(|dim| dim.as_u64().unwrap())
    //             .collect();
    //         let offsets = info["data_offsets"].as_array().unwrap();
    //         let start = offsets[0].as_u64().unwrap() as usize;
    //         let end = offsets[1].as_u64().unwrap() as usize;
    //         let raw = &data[start..end];
    //         let mut count: u64 = 1;
    //         for dim in &shape {
    //             count *= dim;
    //         }
    //         let take = count.min(4) as usize;
    //         let mut first = Vec::new();
    //         for i in 0..take {
    //             let bits = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    //             let value = if dtype == "F32" {
    //                 f32::from_bits(bits) as f64
    //             } else if dtype == "I32" {
    //                 bits as i32 as f64
    //             } else {
    //                 eprintln!("unsupported dtype {dtype}");
    //                 process::exit(1);
    //             };
    //             first.push(value);
    //         }
    //         tensors.push(serde_json::json!({
    //             "name": name,
    //             "dtype": dtype,
    //             "shape": shape,
    //             "first": first,
    //         }));
    //     }
    //     println!("{}", serde_json::to_string(&tensors).unwrap());
    // } else {
    //     eprintln!("Unknown command {}", args[1]);
    //     process::exit(1);
    // }
}

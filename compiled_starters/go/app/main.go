package main

import (
	// TODO: Uncomment the code below to pass the first stage!
	// "encoding/binary"
	// "encoding/json"
	// "io"
	// "math"
	// "sort"
	"fmt"
	"os"
)

// Usage: your_program.sh <command> <arg1> <arg2> ...
func main() {
	// You can use print statements as follows for debugging, they'll be visible when running tests.
	fmt.Fprintf(os.Stderr, "Logs from your program will appear here!\n")

	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: your_program.sh <command> [<args>...]\n")
		os.Exit(1)
	}

	switch command := os.Args[1]; command {
	case "inspect":
		// TODO: Uncomment the code below to pass the first stage!
		//
		// f, err := os.Open(os.Args[2])
		// if err != nil {
		// 	fmt.Fprintf(os.Stderr, "Error reading file: %s\n", err)
		// 	os.Exit(1)
		// }
		// defer f.Close()
		//
		// var headerLen uint64
		// if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		// 	fmt.Fprintf(os.Stderr, "Error reading header length: %s\n", err)
		// 	os.Exit(1)
		// }
		// headerBytes := make([]byte, headerLen)
		// if _, err := io.ReadFull(f, headerBytes); err != nil {
		// 	fmt.Fprintf(os.Stderr, "Error reading header: %s\n", err)
		// 	os.Exit(1)
		// }
		// var header map[string]struct {
		// 	DType       string `json:"dtype"`
		// 	Shape       []int  `json:"shape"`
		// 	DataOffsets [2]int `json:"data_offsets"`
		// }
		// if err := json.Unmarshal(headerBytes, &header); err != nil {
		// 	fmt.Fprintf(os.Stderr, "Error parsing header: %s\n", err)
		// 	os.Exit(1)
		// }
		// data, err := io.ReadAll(f)
		// if err != nil {
		// 	fmt.Fprintf(os.Stderr, "Error reading data: %s\n", err)
		// 	os.Exit(1)
		// }
		//
		// names := make([]string, 0, len(header))
		// for name := range header {
		// 	if name == "__metadata__" {
		// 		continue
		// 	}
		// 	names = append(names, name)
		// }
		// sort.Strings(names)
		//
		// summaries := make([]struct {
		// 	Name  string    `json:"name"`
		// 	DType string    `json:"dtype"`
		// 	Shape []int     `json:"shape"`
		// 	First []float64 `json:"first"`
		// }, 0, len(names))
		// for _, name := range names {
		// 	info := header[name]
		// 	raw := data[info.DataOffsets[0]:info.DataOffsets[1]]
		// 	count := 1
		// 	for _, dim := range info.Shape {
		// 		count *= dim
		// 	}
		// 	take := count
		// 	if take > 4 {
		// 		take = 4
		// 	}
		// 	first := make([]float64, take)
		// 	for i := 0; i < take; i++ {
		// 		bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		// 		if info.DType == "F32" {
		// 			first[i] = float64(math.Float32frombits(bits))
		// 		} else if info.DType == "I32" {
		// 			first[i] = float64(int32(bits))
		// 		} else {
		// 			fmt.Fprintf(os.Stderr, "unsupported dtype %s\n", info.DType)
		// 			os.Exit(1)
		// 		}
		// 	}
		// 	summaries = append(summaries, struct {
		// 		Name  string    `json:"name"`
		// 		DType string    `json:"dtype"`
		// 		Shape []int     `json:"shape"`
		// 		First []float64 `json:"first"`
		// 	}{name, info.DType, info.Shape, first})
		// }
		// encoded, err := json.Marshal(summaries)
		// if err != nil {
		// 	fmt.Fprintf(os.Stderr, "Error encoding output: %s\n", err)
		// 	os.Exit(1)
		// }
		// fmt.Println(string(encoded))

	default:
		fmt.Fprintf(os.Stderr, "Unknown command %s\n", command)
		os.Exit(1)
	}
}

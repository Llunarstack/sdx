// Fast JSONL merge with optional dedupe by a record key (first row wins).
// Usage: sdx-manifest merge -o out.jsonl [--dedupe-key image_path] a.jsonl b.jsonl ...
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

func imageKey(v map[string]interface{}) string {
	for _, k := range []string{"image_path", "path", "image"} {
		if s, ok := v[k].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

func merge(outPath, dedupeKey string, inputs []string) error {
	seen := map[string]struct{}{}
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()

	for _, path := range inputs {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		sc := bufio.NewScanner(f)
		// Large lines
		buf := make([]byte, 0, 64*1024)
		sc.Buffer(buf, 1024*1024)
		for sc.Scan() {
			line := sc.Text()
			if line == "" {
				continue
			}
			var obj map[string]interface{}
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				continue
			}
			var key string
			if dedupeKey != "" {
				if s, ok := obj[dedupeKey].(string); ok && s != "" {
					key = s
				}
			}
			if key == "" {
				key = imageKey(obj)
			}
			if key == "" {
				continue
			}
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			raw, err := json.Marshal(obj)
			if err != nil {
				continue
			}
			if _, err := w.Write(append(raw, '\n')); err != nil {
				f.Close()
				return err
			}
		}
		if err := sc.Err(); err != nil {
			f.Close()
			return err
		}
		f.Close()
	}
	return nil
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "merge":
		runMerge(os.Args[2:])
		return
	case "explore-stats":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: sdx-manifest explore-stats manifest.jsonl")
			os.Exit(2)
		}
		if err := exploreStats(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	case "explore-dedupe":
		runExploreDedupe(os.Args[2:])
		return
	default:
		printUsage()
		os.Exit(2)
	}
}

func printUsage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  sdx-manifest merge -o out.jsonl [--dedupe-key image_path] a.jsonl ...")
	fmt.Fprintln(os.Stderr, "  sdx-manifest explore-stats explore_manifest.jsonl")
	fmt.Fprintln(os.Stderr, "  sdx-manifest explore-dedupe -o out.jsonl [--key style_genome_id] in.jsonl")
}

func runExploreDedupe(args []string) {
	fs := flag.NewFlagSet("explore-dedupe", flag.ExitOnError)
	out := fs.String("o", "", "output JSONL")
	key := fs.String("key", "style_genome_id", "dedupe field")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if *out == "" || len(fs.Args()) < 1 {
		fmt.Fprintln(os.Stderr, "usage: sdx-manifest explore-dedupe -o out.jsonl [--key style_genome_id] in.jsonl")
		os.Exit(2)
	}
	if err := exploreDedupe(fs.Args()[0], *out, *key); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runMerge(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: sdx-manifest merge -o out.jsonl [--dedupe-key image_path] a.jsonl b.jsonl ...")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("merge", flag.ExitOnError)
	out := fs.String("o", "", "output JSONL path")
	dedupe := fs.String("dedupe-key", "image_path", "field for dedupe (empty = image_path/path/image only)")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if *out == "" {
		fmt.Fprintln(os.Stderr, "-o required")
		os.Exit(2)
	}
	inputs := fs.Args()
	if len(inputs) < 1 {
		fmt.Fprintln(os.Stderr, "need at least one input .jsonl")
		os.Exit(2)
	}
	dk := *dedupe
	if dk == "" {
		dk = "image_path"
	}
	if err := merge(*out, dk, inputs); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

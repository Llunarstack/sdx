package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

func exploreStats(path string) error {
	type acc struct {
		byGenome   map[string]int
		byKind     map[string]int
		total      int
		uniqueCaps int
	}

	a := acc{byGenome: map[string]int{}, byKind: map[string]int{}}
	captions := map[string]struct{}{}

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
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
		a.total++
		gid, _ := obj["style_genome_id"].(string)
		if gid == "" {
			gid = "(none)"
		}
		a.byGenome[gid]++
		kind, _ := obj["candidate_kind"].(string)
		if kind == "" {
			kind = "base"
		}
		a.byKind[kind]++
		if cap, ok := obj["caption"].(string); ok && cap != "" {
			captions[cap] = struct{}{}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	a.uniqueCaps = len(captions)

	fmt.Printf("explore_manifest: %s\n", path)
	fmt.Printf("  rows: %d\n", a.total)
	fmt.Printf("  unique captions: %d\n", a.uniqueCaps)
	fmt.Println("  by candidate_kind:")
	kinds := make([]string, 0, len(a.byKind))
	for k := range a.byKind {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		fmt.Printf("    %s: %d\n", k, a.byKind[k])
	}
	fmt.Println("  by style_genome_id:")
	gids := make([]string, 0, len(a.byGenome))
	for k := range a.byGenome {
		gids = append(gids, k)
	}
	sort.Strings(gids)
	for _, g := range gids {
		fmt.Printf("    %s: %d\n", g, a.byGenome[g])
	}
	return nil
}

func exploreDedupe(inPath, outPath, key string) error {
	seen := map[string]map[string]interface{}{}
	order := []string{}

	f, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
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
		k := ""
		if s, ok := obj[key].(string); ok {
			k = s
		}
		if k == "" {
			continue
		}
		if _, ok := seen[k]; !ok {
			seen[k] = obj
			order = append(order, k)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}

	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()
	for _, k := range order {
		raw, err := json.Marshal(seen[k])
		if err != nil {
			continue
		}
		if _, err := w.Write(append(raw, '\n')); err != nil {
			return err
		}
	}
	fmt.Printf("deduped %d -> %d rows (%s) -> %s\n", len(order), len(order), key, outPath)
	return nil
}

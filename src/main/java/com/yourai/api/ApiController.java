package com.yourai.api;

import com.yourai.docs.DocumentService;
import com.yourai.ollama.OllamaService;
import com.yourai.search.*;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@RestController
public class ApiController {
    private final VectorStore vectors;
    private final DocumentService documents;
    private final OllamaService ollama;

    public ApiController(VectorStore vectors, DocumentService documents, OllamaService ollama) {
        this.vectors = vectors; this.documents = documents; this.ollama = ollama;
    }

    @GetMapping("/search")
    public Map<String,Object> search(@RequestParam String v,
        @RequestParam(defaultValue="5") int k,
        @RequestParam(defaultValue="cosine") String metric,
        @RequestParam(defaultValue="hnsw") String algo) {
        List<Double> query = Arrays.stream(v.split(",")).map(String::trim).map(Double::parseDouble).toList();
        return (Map<String,Object>) vectors.search(query, k, DistanceMetric.parse(metric), algo).get(0);
    }

    public record InsertRequest(String metadata, String category, List<Double> embedding) {}
    @PostMapping("/insert")
    public Map<String,Object> insert(@RequestBody InsertRequest body) {
        VectorItem item = vectors.insert(body.metadata(), body.category(), body.embedding());
        return Map.of("id", item.id());
    }

    @DeleteMapping("/delete/{id}")
    public Map<String,Object> delete(@PathVariable int id) { return Map.of("ok", vectors.delete(id)); }
    @GetMapping("/items") public List<VectorItem> items() { return vectors.all(); }

    @GetMapping("/benchmark")
    public Map<String,Object> benchmark(@RequestParam String v, @RequestParam(defaultValue="5") int k,
                                        @RequestParam(defaultValue="cosine") String metric) {
        List<Double> q = Arrays.stream(v.split(",")).map(String::trim).map(Double::parseDouble).toList();
        return vectors.benchmark(q, k, DistanceMetric.parse(metric));
    }
    @GetMapping("/hnsw-info") public Map<String,Object> graphInfo() { return vectors.graphInfo(); }
    @GetMapping("/stats") public Map<String,Object> stats() { return vectors.stats(); }

    public record DocumentRequest(String title, String text) {}
    @PostMapping("/doc/insert") public Map<String,Object> insertDocument(@RequestBody DocumentRequest req) {
        return documents.insert(req.title(), req.text());
    }
    @GetMapping("/doc/list") public List<Map<String,Object>> listDocuments() { return documents.list(); }
    @DeleteMapping("/doc/delete/{id}") public Map<String,Object> deleteDocument(@PathVariable int id) {
        return Map.of("ok", documents.delete(id));
    }
    public record QuestionRequest(String question, Integer k) {}
    @PostMapping("/doc/search") public Map<String,Object> documentSearch(@RequestBody QuestionRequest req) {
        if (req.question() == null || req.question().isBlank()) throw new IllegalArgumentException("question is required");
        return Map.of("contexts", documents.retrieve(req.question(), req.k() == null ? 3 : req.k()));
    }
    @PostMapping("/doc/ask") public Map<String,Object> ask(@RequestBody QuestionRequest req) {
        if (req.question() == null || req.question().isBlank()) throw new IllegalArgumentException("question is required");
        int k = req.k() == null ? 3 : req.k();
        Map<String,Object> rag = documents.askWithContext(req.question(), k);
        return Map.of("answer", rag.get("answer"), "contexts", rag.get("contexts"),
            "model", ollama.generateModel(), "docCount", documents.size());
    }
    @GetMapping("/status") public Map<String,Object> status() {
        return Map.of("ollamaAvailable", ollama.available(), "embedModel", ollama.embedModel(),
            "genModel", ollama.generateModel(), "docCount", documents.size(),
            "documentSearchAlgorithm", documents.indexAlgorithm(), "docDims", documents.dimensions(),
            "demoDims", VectorStore.DEMO_DIMS, "demoCount", vectors.size());
    }
}

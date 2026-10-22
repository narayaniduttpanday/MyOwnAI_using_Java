package com.yourai.ollama;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import java.util.*;

@Service
public class OllamaService {
    private final RestClient client;
    private final String embedModel;
    private final String generateModel;

    public OllamaService(@Value("${ollama.base-url}") String baseUrl,
                         @Value("${ollama.embed-model}") String embedModel,
                         @Value("${ollama.generate-model}") String generateModel) {
        this.client = RestClient.builder().baseUrl(baseUrl).build();
        this.embedModel = embedModel;
        this.generateModel = generateModel;
    }

    @SuppressWarnings("unchecked")
    public List<Double> embed(String text) {
        Map<String,Object> body = Map.of("model", embedModel, "input", text);
        Map<String,Object> response = client.post().uri("/api/embed").contentType(MediaType.APPLICATION_JSON)
            .body(body).retrieve().body(Map.class);
        if (response == null) return List.of();
        Object embeddings = response.get("embeddings");
        if (embeddings instanceof List<?> list && !list.isEmpty() && list.get(0) instanceof List<?> vector) {
            return vector.stream().map(x -> ((Number)x).doubleValue()).toList();
        }
        Object legacy = response.get("embedding");
        if (legacy instanceof List<?> vector) return vector.stream().map(x -> ((Number)x).doubleValue()).toList();
        return List.of();
    }

    @SuppressWarnings("unchecked")
    public String generate(String prompt) {
        Map<String,Object> response = client.post().uri("/api/generate")
            .contentType(MediaType.APPLICATION_JSON)
            .body(Map.of("model", generateModel, "prompt", prompt, "stream", false))
            .retrieve().body(Map.class);
        return response == null ? "" : String.valueOf(response.getOrDefault("response", ""));
    }

    public boolean available() {
        try { client.get().uri("/api/tags").retrieve().toBodilessEntity(); return true; }
        catch (Exception ex) { return false; }
    }

    public String embedModel() { return embedModel; }
    public String generateModel() { return generateModel; }
}

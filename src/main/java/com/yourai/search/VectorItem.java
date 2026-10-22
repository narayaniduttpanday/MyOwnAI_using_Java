package com.yourai.search;

import java.util.List;

public record VectorItem(int id, String metadata, String category, List<Double> embedding) {
    public VectorItem {
        if (metadata == null) metadata = "";
        if (category == null) category = "uncategorized";
        if (embedding == null || embedding.isEmpty()) throw new IllegalArgumentException("embedding is required");
        embedding = List.copyOf(embedding);
    }
}

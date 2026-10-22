package com.yourai.docs;

import com.yourai.ollama.OllamaService;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class DocumentServiceTest {
    @Test void documentEmbeddingDimensionIsDetectedDynamicallyAndRagReturnsContexts() {
        OllamaService ollama = mock(OllamaService.class);
        when(ollama.embed(anyString())).thenReturn(List.of(1.0, 0.0, 0.0));
        when(ollama.generate(anyString())).thenReturn("This is the generated answer.");
        DocumentService service = new DocumentService(ollama);

        Map<String,Object> inserted = service.insert("Test document", "This is a small test document.");
        assertEquals(3, inserted.get("dims"));
        assertEquals(3, service.dimensions());
        List<Map<String,Object>> contexts = service.retrieve("What is this?", 3);
        assertEquals(1, contexts.size());
        assertEquals("Test document", contexts.get(0).get("title"));

        Map<String,Object> rag = service.askWithContext("What is this?", 3);
        assertEquals("This is the generated answer.", rag.get("answer"));
        assertEquals(1, ((List<?>) rag.get("contexts")).size());
        verify(ollama, times(1)).generate(anyString());
    }

    @Test void switchesToHnswAtTenChunksAndRebuildsAfterDelete() {
        OllamaService ollama = mock(OllamaService.class);
        when(ollama.embed(anyString())).thenReturn(List.of(1.0, 0.0, 0.0));
        DocumentService service = new DocumentService(ollama);
        for (int i = 0; i < 10; i++) service.insert("doc-" + i, "Document number " + i + " contains sample text.");
        assertEquals("hnsw", service.indexAlgorithm());
        assertEquals(10, service.size());
        assertTrue(service.delete(10));
        assertEquals("bruteforce", service.indexAlgorithm());
    }

    @Test void rejectsEmptyDocumentInput() {
        DocumentService service = new DocumentService(mock(OllamaService.class));
        assertThrows(IllegalArgumentException.class, () -> service.insert("", "text"));
    }
}

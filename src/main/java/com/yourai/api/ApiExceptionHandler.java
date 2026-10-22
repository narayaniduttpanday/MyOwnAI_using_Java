package com.yourai.api;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestControllerAdvice
public class ApiExceptionHandler {
    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String,String> badRequest(IllegalArgumentException ex) { return Map.of("error", ex.getMessage()); }

    @ExceptionHandler(IllegalStateException.class)
    @ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
    public Map<String,String> unavailable(IllegalStateException ex) { return Map.of("error", ex.getMessage()); }
}

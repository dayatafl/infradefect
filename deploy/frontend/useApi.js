import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// In production: REACT_APP_API_URL = your HuggingFace Spaces URL
// In development: falls back to localhost:8000 (via package.json proxy)
const BASE_URL = process.env.REACT_APP_API_URL || '';

const API = axios.create({ baseURL: BASE_URL });

export function useHealth() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const check = useCallback(async () => {
    try {
      const { data } = await API.get('/api/health');
      setStatus(data);
    } catch {
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, [check]);

  return { status, loading, refetch: check };
}

export function usePredict() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const { data } = await API.post('/api/predict', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Is the API running?');
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, predict, reset };
}

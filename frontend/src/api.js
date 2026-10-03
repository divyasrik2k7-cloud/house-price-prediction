import axios from 'axios';

const baseURL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export const api = axios.create({
  baseURL,
  timeout: 60000,
});

export function friendlyError(error) {
  if (error?.response?.data?.detail) {
    const detail = error.response.data.detail;
    return typeof detail === 'string' ? detail : JSON.stringify(detail);
  }

  if (error?.code === 'ECONNABORTED') {
    return 'The request timed out. Please try again.';
  }

  if (error?.message === 'Network Error') {
    return 'Cannot reach the backend. Make sure FastAPI is running at ' + baseURL + '.';
  }

  return error?.message || 'An unexpected error occurred.';
}

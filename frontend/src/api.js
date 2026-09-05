import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
  timeout: 15000,
});

export const getDashboard = () => api.get("/dashboard").then(r => r.data);
export const getPayments = () => api.get("/payments").then(r => r.data);
export const getPayment = (id) => api.get(`/payments/${id}`).then(r => r.data);
export const getCases = () => api.get("/recovery-cases").then(r => r.data);
export const analyzeCase = (id) => api.post(`/recovery-cases/${id}/analyze`).then(r => r.data);
export const executeCase = (id) => api.post(`/recovery-cases/${id}/execute`).then(r => r.data);
export const getAnalytics = () => api.get("/analytics").then(r => r.data);

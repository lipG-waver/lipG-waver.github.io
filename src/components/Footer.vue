<template>
  <footer>
    <p>&copy; 2025 LipG. All rights reserved.</p>
  </footer>
</template>

<script>
export default {
  name: 'AppFooter'
}
</script>

<style scoped>
footer {
  margin-top: 60px;
  text-align: center;
  font-size: 1rem;
  color: #7f8c8d;
  padding: 25px 0;
  background: linear-gradient(135deg, #2c3e50, #1a2a3a);
  border-top: 1px solid #eee;
  box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
}

footer p {
  margin: 0;
  color: #ecf0f1;
  font-weight: 300;
  letter-spacing: 0.5px;
}

footer p::before {
  content: "© ";
}

@media (max-width: 768px) {
  footer {
    margin-top: 40px;
    padding: 20px 0;
    font-size: 0.9rem;
  }
}
</style>
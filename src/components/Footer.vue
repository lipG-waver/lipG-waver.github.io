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
  color: #495057;
  padding: 25px 0;
  background: linear-gradient(135deg, #ffffff, #f8f9fa);
  border-top: 1px solid #e9ecef;
  box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.05);
}

footer p {
  margin: 0;
  color: #212529;
  font-weight: 400;
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
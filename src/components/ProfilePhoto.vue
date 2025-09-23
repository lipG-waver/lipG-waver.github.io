<template>
  <img src="../assets/your-photo.jpg" alt="LipG" class="profile-photo" />
</template>

<script>
export default {
  name: 'ProfilePhoto'
}
</script>

<style scoped>
.profile-photo {
  width: 160px;
  height: 160px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  margin-bottom: 25px;
  border: 3px solid #fff;
  transition: all 0.3s ease;
  position: relative;
}

.profile-photo::before {
  content: '';
  position: absolute;
  top: -5px;
  left: -5px;
  right: -5px;
  bottom: -5px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3498db, #2c3e50);
  z-index: -1;
  animation: rotate 3s linear infinite;
}

.profile-photo:hover {
  transform: scale(1.05) rotate(2deg);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 768px) {
  .profile-photo {
    width: 130px;
    height: 130px;
    margin-bottom: 20px;
  }
}
</style>
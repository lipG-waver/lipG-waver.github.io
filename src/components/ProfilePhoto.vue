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
  width: 150px;
  height: 150px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  margin-bottom: 20px;
}
</style>
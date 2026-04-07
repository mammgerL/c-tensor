<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

const navItems = [
  { path: '/', label: '首页', name: '首页' },
  { path: '/playground', label: '计算演示', name: '计算演示' },
  { path: '/training', label: '训练过程', name: '训练过程' },
  { path: '/learn', label: 'C 代码原理', name: 'C 代码原理' },
  { path: '/explore', label: '数据探索', name: '数据探索' },
]

const isActive = (path) => route.path === path

function navigate(path) {
  router.push(path)
}
</script>

<template>
  <nav class="navbar">
    <div class="nav-brand" @click="navigate('/')">
      <span class="nav-title">C-Tensor</span>
    </div>
    <div class="nav-links">
      <button
        v-for="item in navItems"
        :key="item.path"
        :class="['nav-link', { active: isActive(item.path) }]"
        @click="navigate(item.path)"
      >
        {{ item.label }}
      </button>
    </div>
  </nav>
</template>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 32px;
  background: var(--color-card);
  box-shadow: var(--shadow-sm);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
}

.nav-title {
  font-size: 22px;
  font-weight: 800;
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.nav-links {
  display: flex;
  gap: 8px;
}

.nav-link {
  padding: 10px 20px;
  border-radius: var(--radius-sm);
  font-size: 16px;
  font-weight: 600;
  background: transparent;
  color: var(--color-text-light);
}

.nav-link:hover {
  background: var(--color-bg);
  color: var(--color-primary);
}

.nav-link.active {
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: white;
}
</style>

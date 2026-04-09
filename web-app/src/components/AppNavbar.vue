<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const repoUrl = 'https://github.com/mammgerL/c-tensor'

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

  <a
    class="repo-link"
    :href="repoUrl"
    target="_blank"
    rel="noreferrer"
    aria-label="GitHub 仓库"
  >
    <svg class="repo-icon" viewBox="0 0 16 16" aria-hidden="true">
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
        0-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95
        0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.6 7.6 0 0 1 8 4.84c.68 0 1.36.09 2 .27
        1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
        0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
      />
    </svg>
    <span>GitHub</span>
  </a>
</template>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
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
  flex-wrap: wrap;
  justify-content: flex-end;
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

.repo-link {
  position: fixed;
  top: 14px;
  right: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 15px;
  border-radius: 999px;
  background: #24292e;
  color: white;
  text-decoration: none;
  white-space: nowrap;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  z-index: 220;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.repo-link:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24);
}

.repo-link span {
  font-size: 14px;
  font-weight: 700;
}

.repo-icon {
  width: 18px;
  height: 18px;
}

@media (max-width: 1100px) {
  .navbar {
    padding-right: 140px;
  }
}

@media (max-width: 760px) {
  .navbar {
    padding: 14px 20px;
    flex-direction: column;
    align-items: stretch;
  }

  .nav-brand {
    justify-content: center;
  }

  .nav-links {
    justify-content: center;
  }

  .repo-link {
    top: 10px;
    right: 12px;
    padding: 9px 13px;
  }
}
</style>

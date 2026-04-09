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
    <div class="nav-main">
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
      <a
        class="repo-link"
        :href="repoUrl"
        target="_blank"
        rel="noreferrer"
        aria-label="GitHub 仓库"
      >
        <span class="repo-mark">GitHub</span>
        <span class="repo-name">mammgerL/c-tensor</span>
      </a>
    </div>
  </nav>
</template>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
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

.nav-main {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 14px;
  min-width: 0;
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
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(45, 52, 54, 0.14);
  background: rgba(255, 255, 255, 0.92);
  color: var(--color-text);
  text-decoration: none;
  white-space: nowrap;
  transition: var(--transition);
}

.repo-link:hover {
  border-color: rgba(45, 52, 54, 0.28);
  box-shadow: 0 4px 12px rgba(45, 52, 54, 0.08);
  transform: translateY(-1px);
}

.repo-mark {
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--color-text-light);
}

.repo-name {
  font-size: 14px;
  font-weight: 700;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

@media (max-width: 1100px) {
  .navbar {
    align-items: flex-start;
  }

  .nav-main {
    flex-direction: column;
    align-items: flex-end;
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

  .nav-main {
    align-items: stretch;
  }

  .nav-links {
    justify-content: center;
  }

  .repo-link {
    align-self: center;
  }
}
</style>

import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { createPinia } from 'pinia'
import App from './App.vue'
import './style.css'

const routes = [
  { path: '/', component: () => import('./views/HomeView.vue') },
  { path: '/learn', component: () => import('./views/LearnView.vue') },
  { path: '/playground', component: () => import('./views/PlaygroundView.vue') },
  { path: '/explore', component: () => import('./views/ExploreView.vue') },
]

const router = createRouter({
  history: createWebHistory('/c-tensor/'),
  routes,
})

// Handle SPA redirect from 404.html
const redirectPath = sessionStorage.getItem('spa-redirect')
if (redirectPath) {
  sessionStorage.removeItem('spa-redirect')
  router.replace(redirectPath)
}

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Tech Notes',
  description: '技术学习笔记',
  
  // 如果部署到 https://<username>.github.io/<repo>/
  // 需要设置 base 为 /<repo>/
  base: '/tech-notes/',
  
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'Linux', link: '/linux/' },
      { text: 'DevOps', link: '/devops/' },
      { text: 'LLM', link: '/LLM-fundamentals/' },
      { text: 'Agent', link: '/agent/' }
    ],

    sidebar: {
      '/agent/': [
        {
          text: 'Agent 开发',
          collapsed: false,
          items: [
            { text: '概览', link: '/agent/' }
          ]
        },
        {
          text: 'LangChain/LangGraph',
          collapsed: false,
          items: [
            { text: '01-环境搭建', link: '/agent/langchain/01-环境搭建' },
            { text: '02-Agent核心概念', link: '/agent/langchain/02-Agent核心概念' },
            { text: '03-快速入门完整示例', link: '/agent/langchain/03-快速入门完整示例' },
            { text: '04-Agent详解', link: '/agent/langchain/04-Agent详解' },
            { text: '05-Models详解', link: '/agent/langchain/05-Models详解' },
            { text: '06-Messages详解', link: '/agent/langchain/06-Messages详解' },
            { text: '07-Tools详解', link: '/agent/langchain/07-Tools详解' },
            { text: '08-记忆系统详解', link: '/agent/langchain/08-记忆系统详解' },
            { text: '09-输出控制详解', link: '/agent/langchain/09-输出控制详解' },
            { text: '10-中间件详解', link: '/agent/langchain/10-中间件详解' },
            { text: '11-Guardrails', link: '/agent/langchain/11-Guardrails' },
            { text: '12-Runtime详解', link: '/agent/langchain/12-Runtime详解' },
            { text: '13-上下文工程详解', link: '/agent/langchain/13-上下文工程详解' },
            { text: '14-MCP详解', link: '/agent/langchain/14-MCP详解' },
            { text: '15-HITL详解', link: '/agent/langchain/15-HITL详解' },
            { text: '16-多Agent架构详解', link: '/agent/langchain/16-多Agent架构详解' },
            { text: '17-LangSmith工具链详解', link: '/agent/langchain/17-LangSmith工具链详解' },
            { text: '18-RAG检索详解', link: '/agent/langchain/18-RAG检索详解' },
            { text: '19-Deep Research实践', link: '/agent/langchain/19-Deep Research实践' }
          ]
        }
      ],
      '/linux/': [
        {
          text: 'Linux',
          collapsed: false,
          items: [
            { text: '概览', link: '/linux/' },
            { text: 'Cgroup资源控制', link: '/linux/Linux-Cgroup资源控制' },
            { text: 'Memory Cgroup详解', link: '/linux/Linux-Memory-Cgroup详解' },
            { text: '容器化技术', link: '/linux/Linux容器化技术' },
            { text: '性能统计原理', link: '/linux/Linux性能统计原理' },
            { text: '性能观测与内核跟踪', link: '/linux/Linux性能观测与内核跟踪' },
            { text: '用户态协程', link: '/linux/Linux用户态协程' },
            { text: '进程调度深度解析', link: '/linux/Linux进程调度深度解析' }
          ]
        }
      ],
      '/devops/': [
        {
          text: 'DevOps',
          collapsed: false,
          items: [
            { text: 'Docker与K8s部署', link: '/devops/Docker与K8s部署学习笔记' },
            { text: 'Git与GitHub使用指南', link: '/devops/Git与GitHub使用指南' }
          ]
        }
      ],
      '/LLM-fundamentals/': [
        {
          text: 'LLM 基础',
          collapsed: false,
          items: [
            { text: '概览', link: '/LLM-fundamentals/' },
            { text: 'NLP基础概念', link: '/LLM-fundamentals/第1章-NLP基础概念' },
            { text: 'Transformer架构', link: '/LLM-fundamentals/第2章-Transformer架构' },
            { text: '预训练语言模型PLM', link: '/LLM-fundamentals/第3章-预训练语言模型PLM' },
            { text: 'LLM应用与进阶', link: '/LLM-fundamentals/第4章-LLM应用与进阶' }
          ]
        },
        {
          text: 'RAG 专题',
          collapsed: true,
          items: [
            { text: 'RAG流程', link: '/LLM-fundamentals/RAG/RAG流程' },
            { text: '01-Embedding原理', link: '/LLM-fundamentals/RAG/01-Embedding原理详解' },
            { text: '02-ANN算法', link: '/LLM-fundamentals/RAG/02-ANN近似最近邻算法详解' },
            { text: '03-向量相似度度量', link: '/LLM-fundamentals/RAG/03-向量相似度度量详解' },
            { text: '04-BM25算法', link: '/LLM-fundamentals/RAG/04-BM25算法详解' },
            { text: '05-倒排索引', link: '/LLM-fundamentals/RAG/05-倒排索引详解' },
            { text: '06-Bi-encoder与Cross-encoder', link: '/LLM-fundamentals/RAG/06-Bi-encoder与Cross-encoder详解' },
            { text: '07-对比学习', link: '/LLM-fundamentals/RAG/07-对比学习详解' },
            { text: '08-Chunking分片策略', link: '/LLM-fundamentals/RAG/08-Chunking分片策略详解' },
            { text: '09-Query改写与扩展', link: '/LLM-fundamentals/RAG/09-Query改写与扩展详解' },
            { text: '10-混合检索与融合策略', link: '/LLM-fundamentals/RAG/10-混合检索与融合策略详解' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wanglei99999/tech-notes' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: '持续学习，持续记录',
      copyright: 'Copyright © 2025-present'
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新于'
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题'
  },

  markdown: {
    math: true
  },

  lastUpdated: true
})

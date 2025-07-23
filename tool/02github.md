---
title: github push request
date: 2025-07-23
updated: 2025-07-23
description:
---

## 找项目

这个很简单：确定目标项目，此处以网页教程提供为例：
`https://github.com/leerob/next-mdx-blog`

找到目标项目之后，可以先查看issue和PR中，是否存在已提出的自己需要解决的问题，或者自己提出问题，等待他人解决。

![fix](/images/image-2-1.png)

如果没有人回应，再尝试自己提交 PR。

![fix](/images/image-1-1.png)

## 派生存储库（Forking a repository）

进入`https://github.com/leerob/next-mdx-blog`项目主页，点击**Fork**按钮，创建一个新的派生项目：

![fix](/images/image-3-1.png)

命名派生项目并创建：

![fix](/images/image-4-1.png)

## 克隆一个派生项目（cloning a fork）

进入自己的 Github 工作区，将派生项目克隆到本地（或者远端服务器）：

![fix](/images/image-5-1.png)

随后在终端输入命令 `git clone https://github.com/edisonchan1/owe-me-300yuan-s-github-pr-.git`
即可在当前文件目录下克隆派生项目项目

![fix](/images/image-6-1.png)

![fix](/images/image-7-1.png)

## 创建一个分支（Creating a branch）

在终端输入命令`git checkout -b +name`（可参考同目录的文件 `01git.md`查看详细git命令介绍）

## 作出修改（Making changes）

删除一行无用的代码`import Link from 'next/l`

![fix](/images/image-8-1.png)

## 提交修改（Pushing changes）

当作者新的提交和PR被合并到原项目，在你的工作去的派生项目会显示原项目有更新。例如

![fix](/images/image-10-1.png)

点击`Update branch`之后，将原项目更新同步到派生项目，再进入本地项目文件夹`cd + name`

![fix](/images/image-11-1.png)

这里有三行命令：
`git stash`：把当前尚未提交的工作临时保存起来，从而让你能够切换到其他分支或者开展其他操作，而不会丢失当前的修改内容。
`git pusll`：从远程仓库获取最新提交并将其合并到当前本地分支的 Git 命令
`git stash pop`：用于恢复之前暂存的工作状态的命令

## 创建合并请求（Creating a pull request）

使用命令`git add .`
修改本地克隆的派生项目后，提交可修改`git commit -m"changes"`
推送到远程仓库`git push -u origin + name`

![fix](/images/image-12-1.png)

此时就可以在GitHub网站上收到这样的通知。

![fix](/images/image-13-1.png)

点击 Create pull request ，就行了。不出意外，你提的 PR 就应该躺在下面了：

![fix](/images/image-14-1.png)

## 删除你的分支（Deleting your branch）

最后一步不是必须的，只是保持一个规范的开源协作习惯，减少意外提交错误项目分支的情况发生。

来到原项目 Github 主页，找到之前已经合并的提交请求（在关闭的 PR 列表中），点击 Delete branch

![fix](/images/image-15-1.png)

删除本地分支：

`git branch -d feature/new-feature`

注意：需要先切换到主分支。

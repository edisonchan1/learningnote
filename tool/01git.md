## 基础命令

`git add + file`：跟踪
`git commit -m'注释'`：提交修改
`git status`：查看修改状态
`git diff + file`：查看修改内容
`git log`：显示日志
`git log --pretty=oneline`：显示日志（--pretty=oneline无空格）
`git reset HEAD^`：返回上一个版本 （^^返回上上个版本）
`git reflog`：查看所有版本号
`git reset --hard 22a913a`：返回指定序号的版本

## 工作区与暂存区的区别

工作区：就是你在电脑上看到的目录，比如目录下testgit里的文件(.git隐藏目录版本库除外)。或者以后需要再新建的目录文件等等都属于工作区范畴。
版本库(Repository)：工作区有一个隐藏目录.git,这个不属于工作区，这是版本库。其中版本库里面存了很多东西，其中最重要的就是stage(暂存区)，还有Git为我们自动创建了第一个分支master,以及指向master的一个指针HEAD。

Git提交文件到版本库有两步：

1. 是使用git add把文件添加进去，实际上就是把文件添加到暂存区。
2. 使用git commit提交更改，实际上就是把暂存区的所有内容提交到当前分支。

演示如下：
我们在readme.txt再添加一行内容为4444444，接着在目录下新建一个文件为test.txt 内容为test，我们先用命令 git status来查看下状态，如下：![fix](/images/image.png)

现在使用git add 命令把两个文件都添加到暂存区中，使用git commit一次性提交到分支，并且使用git status 查看前后两次都状态：
![fix](/images/image-1.png)
![fix](/images/image-2.png)

## Git撤销修改和删除文件操作

### 撤销修改

现在在readme.txt文件内添加内容5555555555。查看如下：
![fix](/images/image-3.png)

如果发现有误，需要恢复以前的版本。根据基础操作有如下两种方法：

1. 手动更改去掉需要修改的内容，再使用`add`、 `commit`命令。
2. 直接恢复到上个版本：`git reset --hard HEAD^`

这里介绍另一种方法：先使用`git status`查看当前状态：
![fix](/images/image-4.png)
接下来使用`git checkout -- readme.txt`(中间有空格)并查看修改后的内容如下：
![fix](/images/image-5.png)

命令 `git checkout --readme.txt` 意思就是，把readme.txt文件在工作区做的修改全部撤销，这里有2种情况，如下：

1. readme.txt自动修改后，还没有放到暂存区，使用 撤销修改就回到和版本库一模一样的状态。
2. 另外一种是readme.txt已经放入暂存区了，接着又作了修改，撤销修改就回到添加暂存区后的状态

上文介绍的第一种情况，第二种情况如下：
假如现在我对readme.txt添加一行 内容为6666666666666，我`git add` 增加到暂存区后，接着添加内容7777777，我想通过撤销命令让其回到暂存区后的状态。如下所示：
![fix](/images/image-6.png)

### 删除文件

假如在testgit目录中国呢添加了一个文件b.txt然后提交：
![fix](/images/image-7.png)
可以直接从文件目录删除文件。如果想彻底从版本库中删除文件，可以再执行`commit`命令。

可以使用`git checkout -- b.txt`
![fix](/images/image-8.png)

这时b.txt又出现在了目录中。

## 远程仓库

### 创建新仓库

![fix](/images/image-10.png)

设置仓库名字与选项后创建仓库并复制地址
![fix](/images/image-11.png)

在本地的testgit仓库运行命令：
`git remote add origin https://github.com/tugenhua0707/testgit`
`git push -u origin main`（第一次要加上 -u 参数）
返回如下
![fix](/images/image-12.png)

返回页面并刷新便上传成功：
![fix](/images/image-13.png)

之后只要本地作了提交，就可以通过如下命令：
`git push origin main`
把本地的main分支的最新修改推送到GitHub上。现在就拥有了真正的分布式版本库。

### 克隆远程仓库

接下来继续创建一个新的远程库：![fix](/images/image-14.png)
勾选`initialize this repository with a README`会自动生成一个README.md文件。

![fix](/images/image-15.png)
现在远程库已经准备好了，下一步是准备命令`git clone`克隆一个本地库了。
![fix](/images/image-17.png)
克隆结果如下：
![fix](/images/image-16.png)

## 分支

### 创建与合并分支

首先，我们来创建dev分支，然后切换到dev分支上。如下操作：
![fix](/images/image-18.png)
`git checkout -b dev`加上 -b 参数表示创建并切换，相当于如下两天命令

1. git branch dev
2. git checkout dev

`git branch`查看分支，会列出所有分支，当前分支前面会添加一个*号
![fix](/images/image-19.png)
在dev分支上继续编辑，如在readme.txt再添加一行777777777
首先查看readme.txt的内容，接着添加内容777777777并保存如下：
![fix](/images/image-21.png)

分支dev修改完成，接下来切换到主分支main上，继续查看readme.txt内容如下：
![fix](/images/image-22.png)

现在把dev分支上的内容合并到分支main上，可以在main分支上使用如下命令
`git merge dev`
![fix](/images/image-23.png)

可以看到合并后的两个分支完全一样。

这里是Fast-forward信息，Git告诉我们，这次合并是“快速模式”，也就是直接把main指向dev的当前提交，合并速度很快。

合并完成后删除dev分支操作如下：![fix](/images/image-24.png)

分支命令总结如下：

1. 查看分支：`git branch`
2. 创建分支：`git branch name`
3. 切换分支：`git checkout name`
4. 创建+切换分支：`git chechout -b name`
5. 合并某分支到当前分支：`git merge name`
6. 删除分支：`git branch -d name`

### 解决分支冲突

先创建一个新分支，命名为fenzhi1，添加新的一行888888888
再切换到主分支main，添加新的一行999999999
![fix](/images/image-25.png)

现在需要在main分支上合并fenzhi1，操作如下：

使用命令`cat readme.txt`结果如下
![fix](/images/image-26.png)

Git用<<<<<<<，=======，>>>>>>>标记出不同分支的内容，
其中<<<<<<< HEAD：是指主分支修改的内容，
>>>>>>> fenzhi1：是指fenzhi1上修改的内容，我们可以修改下如下后保存：
![fix](/images/image-27.png)手动修改后保存。

使用`git log`查看修改日志：
![fix](/images/image-28.png)

### 分支管理策略

合并分支时，git一般使用“Fast-forward”模式，在这种模式下，删除分支后，会丢掉分支信息，现在我们来使用带参数-no-ff来禁用“Fast-forward”模式。首先我们来做demo演示下：

1. 创建一个dev分支
2. 修改readme.txt内容
3. 添加到暂存区
4. 切换回主分支
5. 合并dev分支，使用`git merge -no-ff -m`来“注释” dev
6. 查看历史记录

![fix](/images/image-29.png)

## bug 分支

开发中经常会碰到bug问题，有了bug就要修复，在Git中，分支是很强大的，每个bug都可以通过一个临时分支来修复，修复完成后，合并分支，然后将临时的分支删除掉。
当出现问题时，可使用`git stash`隐藏问题，进而创建分支，在分支中修正bug
![fix](/images/image-31.png)
这里介绍issue-404修复bug。
![fix](/images/image-30.png)
![fix](/images/image-32.png)

修复完成后，切换到main分支上，并完成合并，最后删除issue-404分支。

## 多人协作

从远程库克隆时，实际上Git自动把本地的main分支和远程的main分支对应起来了，并且远程库的默认名称是origin。

 · 要查看远程库信息，使用`git remote`
 · 要查看远程库的详细信息，使用`git remote -v`
![fix](/images/image-33.png)

推送分支就是把该分支上所有本地提交到远程库中，推送时，要指定本地分支，这样，Git就会把该分支推送到远程库对应的远程分支上：使用命令 `git push origin master`

本地的内容如下：
![fix](/images/image-34.png)

使用命令`git push prigin main`上传的内容如下：
![fix](/images/image-35.png)

可以看到 推送成功了，如果我们现在要推送到其他分支，比如dev分支上，我们还是那个命令 git push origin dev那么一般情况下，那些分支要推送呢？master分支是主分支，因此要时刻与远程同步。一些修复bug分支不需要推送到远程去，可以先合并到主分支上，然后把主分支master推送到远程去。

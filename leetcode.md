# Leetcode I

Graph topics to study for interviews:

- BFS
- DFS
- Topological Sort & Shortest-path in a DAG
- Dijkstra's algorithm
- Bellman-Ford
- A-star (A*)
- Floyd-Warshall (debatable, but it's 5 lines of code, so no reason not to know it)

Leetcode problem classification: http://www.programcreek.com/2013/08/leetcode-problem-classification/
 leetcode/lintcode: http://www.kancloud.cn/kancloud/data-structure-and-algorithm-notes/73063
java leetcode code: https://github.com/lydxlx1/LeetCode
Interesting pieces of code and questions: https://sites.google.com/site/spaceofjameschen/ very relevant

Concept #1
Lower bound for any comparison based sorting algorithm :

1) Each of the n! permutations on n elements must appear as one of the leaves of the decision tree for the sorting algorithm to sort properly.

2) Let x be the maximum number of comparisons in a sorting algorithm. The maximum height of the decison tree would be x. A tree with maximum height x has at most 2^x leaves.

After combining the above two facts, we get following relation.

      n!  <= 2^x

Taking Log on both sides.
      log2(n!)  <= x

Since log2(n!)  = Θ(nLogn),  we can say
      x = Ω(nLog2n)




Problems to revise:

- Reverse Binary Tree Level Order Traversal :

-     3
  / \
  9  20
    /  \
  15  7


return its bottom-up level order traversal as:
[
  [15,7],
  [9,20],
  [3]
]

```java
public class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<List<Integer>> wrapList = new LinkedList<List<Integer>>();
        
        if(root == null) return wrapList;
        
        queue.offer(root);
        while(!queue.isEmpty()){
            int levelNum = queue.size();
            List<Integer> subList = new LinkedList<Integer>();
            for(int i=0; i<levelNum; i++) {
                if(queue.peek().left != null) queue.offer(queue.peek().left);
                if(queue.peek().right != null) queue.offer(queue.peek().right);
                subList.add(queue.poll().val);
            }
            wrapList.add(0, subList);
        }
        return wrapList;
    }}
```

Using DFS:
```java
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> list = new LinkedList<>();
        dfs(root, list, 0);
        return list;
    }
    private void dfs(TreeNode root, List<List<Integer>> list, int level) {
         if(root == null) return;
         if(list.size() == level) {
             list.add(0, new ArrayList<>());
         }
         dfs(root.left, list, level+1);
         dfs(root.right, list, level+1);
         list.get(list.size()-level-1).add(root.val);
    } 
```
Regular level order using DFS:
```java
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        dfs(root, list, 0);
        return list;
    }
    private void dfs(TreeNode root, List<List<Integer>> list, int level) {
         if(root == null) return;
         if(list.size() == level) {
             list.add(new ArrayList<>());
         }
         dfs(root.left, list, level+1);
         dfs(root.right, list, level+1);
         list.get(level).add(root.val); // you can move this above recursive calls
    }
```

- Majority element II : http://www.geeksforgeeks.org/given-an-array-of-of-size-n-finds-all-the-elements-that-appear-more-than-nk-times/

n/3 times in an array:
```python
class Solution:
# @param {integer[]} nums# @return {integer[]}
def majorityElement(self, nums):
    if not nums:
        return []
    count1, count2, candidate1, candidate2 = 0, 0, 0, 1
    for n in nums:
        if n == candidate1:
            count1 += 1
        elif n == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1, count1 = n, 1
        elif count2 == 0:
            candidate2, count2 = n, 1
        else:
            count1, count2 = count1 - 1, count2 - 1
    return [n for n in (candidate1, candidate2)  if nums.count(n) > len(nums) // 3]
```

- Rotate array : just remember that if k > n , do k %= n
- Set Matrix Zeros  : remember how to separate the case when first row or first column has any zeros. Solution . You can check CTCI solution too
- Add digits: The digital root (also repeated digital sum) of a non-negative integer is the (single digit) value obtained by an iterative process of summing digits, on each iteration using the result from the previous iteration to compute a digit sum. The process continues until a single-digit number is reached. For example, the digital root of 65,536 is 7, because 6 + 5 + 5 + 3 + 6 = 25 and 2 + 5 = 7.  Significance and formula of the digital root[edit] It helps to see the digital root of a positive integer as the position it holds with respect to the largest multiple of 9 less than it. For example, the digital root of 11 is 2, which means that 11 is the second number after 9. Likewise, the digital root of 2035 is 1, which means that 2035 − 1 is a multiple of 9. If a number produces a digital root of exactly 9, then the number is a multiple of 9. With this in mind the digital root of a positive integer n may be defined by using floor function, as
- String to integer: How to detect overflow? Note: always test for (-sys.maxint-1)-1 : Solution
- Partition List: Just insert nodes with values less than x at the end of the list

-  Ugly Number II
``` python
import heapq
class Solution:
    # @param {integer} n
    # @return {integer}
    def nthUglyNumber(self, n):

        ugly_number = 0
        heap = []
        heapq.heappush(heap, 1)
        for _ in xrange(n):
            ugly_number = heapq.heappop(heap)
            if ugly_number % 2 == 0:
                heapq.heappush(heap, ugly_number * 2)
            elif ugly_number % 3 == 0:
                heapq.heappush(heap, ugly_number * 2)
                heapq.heappush(heap, ugly_number * 3)
            else:
                heapq.heappush(heap, ugly_number * 2)
                heapq.heappush(heap, ugly_number * 3)

                heapq.heappush(heap, ugly_number * 5)
        return ugly_number
```
2nd approach:

METHOD 2 (Use Dynamic Programming)
Here is a time efficient solution with O(n) extra space. The ugly-number sequence is 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, …
     because every number can only be divided by 2, 3, 5, one way to look at the sequence is to split the sequence to three groups as below:
     (1) 1×2, 2×2, 3×2, 4×2, 5×2, …
     (2) 1×3, 2×3, 3×3, 4×3, 5×3, …
     (3) 1×5, 2×5, 3×5, 4×5, 5×5, …

We can find that every subsequence is the ugly-sequence itself (1, 2, 3, 4, 5,  …) multiply 2, 3, 5. Then we use similar merge method as merge sort, to get every ugly number from the three subsequence.
Every step we choose the smallest one, and move one step after.
Code:
```python
# good way to avoid duplicates
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        u2, u3, u5 = 0, 0, 0
        ugly_nums = [None for i in range(n)]
        ugly_nums[0] = 1
        for i in range(1, n):
            ugly_nums[i] = min(2*ugly_nums[u2], 3*ugly_nums[u3], 5* ugly_nums[u5])
            if ugly_nums[i] == 2*ugly_nums[u2]:
                u2 += 1
            if ugly_nums[i] == 3*ugly_nums[u3]:
                u3 += 1
            if ugly_nums[i] == 5*ugly_nums[u5]:
                u5 += 1
            #print u2, u3, u5

        return ugly_nums[n-1]
```
-  Merge K sorted lists:

Analysis:
Give an O(n lg k)-time algorithm to merge k sorted lists into one sorted list, where n is the total number of elements in all the input lists.
Solution: The straightforward solution is to pick the smallest of the top elements in each list, repeatedly. This takes k − 1 comparisons per element, in total O(k · n).
Analysis: It takes O(k) to build the heap; for every element, it takes O(lg k) to DeleteMin and O(lg k) to insert the next one from the same list. In total it takes O(k + n lg k) = O(n lg k).
```java
import java.util.*;
public class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<>(lists.length, new Comparator<ListNode>(){
                @Override
                public int compare(ListNode l1, ListNode l2) { return l1.val-l2.val; }
            });

        ListNode mergedList = new ListNode(0);
        ListNode head = mergedList;
        for(int i=0;i<lists.length;i++){
            if(lists[i] != null){
                pq.offer(lists[i]); // we should be able to initialize from a list: PriorityQueue(Collection<? extends E> c)
            }
        }
        while(!pq.isEmpty()){
            ListNode l = pq.poll();
            if(l.next != null){
                pq.offer(l.next);
            }
            mergedList.next = l;
            mergedList = mergedList.next;
        }
        return head.next;
    }
}
```
9. Square root: Start with lo=0 , hi = 1+n/2 for binary search

Leetcode articles:
=============
1. A binary tree problem – Populating next right pointers in each node. Write a function to connect all the adjacent nodes at the same level in a complete binary tree. Won’t work for non-complete binary Tree, use level order traversal(here is my code from leetcode) in that case.

Important to do pre-order traversal here
Ans: From: http://www.geeksforgeeks.org/connect-nodes-at-same-level/
```cpp
struct node {
  int data;
  struct node *left;
  struct node *right;
  struct node *nextRight;};

void connectRecur(struct node* p);

// Sets the nextRight of root and calls connectRecur() for other nodes
void connect (struct node *p)
{
    // Set the nextRight for root
    p->nextRight = NULL;

    // Set the next right for rest of the nodes (other than root)
    connectRecur(p);
}

/* Set next right of all descendents of p.
  Assumption:  p is a compete binary tree */
void connectRecur(struct node* p)
{
  // Base case
  if (!p)    return;

  // Set the nextRight pointer for p's left child
  if (p->left)
    p->left->nextRight = p->right;

  // Set the nextRight pointer for p's right child
  // p->nextRight will be NULL if p is the right most child at its level
  if (p->right)
    p->right->nextRight = (p->nextRight)? p->nextRight->left: NULL;

  // Set nextRight for other nodes in pre order fashion
  connectRecur(p->left);
  connectRecur(p->right);
}

iterative way:

void connect(TreeLinkNode *root) {
    if (root == NULL) return;
    TreeLinkNode *pre = root;
    TreeLinkNode *cur = NULL;
    while(pre->left) {
        cur = pre;
        while(cur) {
            cur->left->next = cur->right;
            if(cur->next) cur->right->next = cur->next->left;
            cur = cur->next;
        }
        pre = pre->left;
    }
}
```
For incomplete binary tree in O(1) space: [https://discuss.leetcode.com/topic/28580/java-solution-with-constant-space]
```java
public void connect(TreeLinkNode root) {
    TreeLinkNode dummyHead = new TreeLinkNode(0);
    TreeLinkNode pre = dummyHead;
    while (root != null) {
            if (root.left != null) {
                    pre.next = root.left;
                    pre = pre.next;
            }
            if (root.right != null) {
                    pre.next = root.right;
                    pre = pre.next;
            }
            root = root.next;
            if (root == null) {
                    pre = dummyHead;
                    root = dummyHead.next;
                    dummyHead.next = null;
            }
    }
}
```
2. Finding intersection of two sorted arrays

Ans: Use O(m+n) approach.
But what if m>>n. Use binary search so as to get O(n log m) time algorithm.

3. Hacking a google interview [MIT]
Refer these and understand the main points from each:
» Handout 1
» Handout 2
» Handout 3
» Common Questions Part 1
» Common Questions Part 2

4. Finding prime numbers:

Ans: The Sieve of Eratosthenes uses an extra O(n) memory and its runtime complexity is O(n log log n). For the more mathematically inclined readers, you can read more about its algorithm complexity on Wikipedia.
```java
/* Generate a prime list from 0 up to n, using The Sieve of Erantosthenes
param n The upper bound of the prime list (including n)
param prime[] An array of truth value whether a number is prime
*/
void prime_sieve(int n, bool prime[]) {
  prime[0] = false;
  prime[1] = false;
  int i;
  for (i = 2; i <= n; i++)
    prime[i] = true;

  int limit = sqrt(n);
  for (i = 2; i <= limit; i++) {
    if (prime[i]) {
      for (int j = i * i; j <= n; j += i)
        prime[j] = false;
    }
  }
}
```
5. Multiplication of numbers in an array of n numbers, compose an array output such that each element is multiplication of all numbers excepts the current number

Ans: Easy approach:
A ->    4,3,2,1,2
```
left[i]->1,4,12,24,24
right[i]->12,4,2,2,1
output[i]->left[i] * right[i]

void array_multiplication(int A[], int OUTPUT[], int n) {
int left = 1;
  int right = 1;
  for (int i = 0; i < n; i++)
  OUTPUT[i] = 1;
  for (int i = 0; i < n; i++) {
    OUTPUT[i] *= left;
    OUTPUT[n - 1 - i] *= right;
    left *= A[i];
    right *= A[n - 1 - i];
  }
}
```
6. Linked list reversal recursive
Ans:
```
def reverseList(self, head):
     if head is None:
          return None
     if head.next is None:
          return head
     l = self.reverseList(head.next)
     head.next.next = head
     head.next = None
     return l

Iterative:
      def reverse(self, head):
          # write your code here
          if not head:
            return head
          prev = None
          curr = head
          while curr:
              temp = curr.next
              curr.next = prev
              prev = curr
              curr = temp
return prev
```
7. Binary Tree Level-Order Traversal Using Depth First Search (DFS)
```
 void printLevel(BinaryTree *p, int level) {   if (!p) return;
  if (level == 1) {
    cout << p->data << " ";
  } else {
    printLevel(p->left, level-1);
    printLevel(p->right, level-1);
  }
}
  void printLevelOrder(BinaryTree *root) {   int height = maxHeight(root);
  for (int level = 1; level <= height; level++) {
    printLevel(root, level);
    cout << endl;
  }
}
```
// Time Complexity: O(n^2) in worst case. For a skewed tree, printGivenLevel() takes O(n) time where n is the number of nodes in the skewed tree. So time complexity of printLevelOrder() is O(n) + O(n-1) + O(n-2) + .. + O(1) which is O(n^2).

8. For finding Maximum Height (Depth) of a Binary Tree iteratively, you can also use BFS level order traversal apart from usual iterative solutions of in, pre or post order traversals

9. Zig zag traversal of binary tree:
```
class Solution:
    # @param root, a tree node
    # @return a list of lists of integers
    def zigzagLevelOrder(self, root):
        if root is None:
            return []
        result, current, level = [], [root], 1
        while current:
            next_level, vals = [], []
            for node in current:
                vals.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            if level % 2:
                result.append(vals)
            else:
                result.append(vals[::-1])
            level += 1
            current = next_level
        return result
```
10. x && (x-1) == 0 if passed 0 would return true; how do you fix it ?

Ans:return x && !(x & (x-1));

11. isBST [accepted on leetcode]
Ans:
```
public class Solution {
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        return isValidBSTRec(root, null, null); // do this to avoid hitting int_max/int_min issues
    }

    public boolean isValidBSTRec(TreeNode root, Integer max, Integer min){
        if(root == null) return true;
        if((max != null && root.val >= max) || (min != null && root.val <= min)) return false; // handle < > here to avoid overflow.
        return isValidBSTRec(root.left, root.val, min) && isValidBSTRec(root.right, max, root.val);
    }
}
```
12. Convert Sorted Array to Balanced Binary Search Tree (BST)
```
 class Solution:
    # @param {integer[]} nums
    # @return {TreeNode}
    def sortedArrayToBST(self, nums):
        n = len(nums)
        if n == 0:
            return None
        root = TreeNode(nums[n/2])
        root.left = self.sortedArrayToBST(nums[:n/2])
        root.right = self.sortedArrayToBST(nums[(n/2)+1:])
        return root
```
13. Find the total area covered by two rectilinear rectangles in a 2D plane. Each rectangle is defined by its bottom left corner and top right corner coordinates.
```
Ans: Cases to consider
a. non intersecting
b. intersecting
c. one rectangle inside another
d. one rectangle as point rectangle
Following code takes care of all these cases:

class Solution:
    # @param {integer} A
    # @param {integer} B
    # @param {integer} C
    # @param {integer} D
    # @param {integer} E
    # @param {integer} F
    # @param {integer} G
    # @param {integer} H
    # @return {integer}
    def computeArea(self, A, B, C, D, E, F, G, H):
        if C<E or G < A:
            return (C-A)*(D-B) + (G-E)*(H-F)
        if D<F or H<B:
            return (C-A)*(D-B) + (G-E)*(H-F)
        right = min(C,G)
        left = max(A,E) # imp
        top = min(D,H)
        bottom =  max(B,F) # imp
        return abs((C-A)*(D-B)) + abs((G-E)*(H-F))- abs((right-left)*(top-bottom))
```
Derived problem: Find if two rectangles overlap or not?
Ans:
```
// Returns true if two rectangles (l1, r1) and (l2, r2) overlap
bool doOverlap(Point l1, Point r1, Point l2, Point r2)
{
    // If one rectangle is on left side of other
    if (l1.x > r2.x || l2.x > r1.x)
        return false;
    // If one rectangle is above other
    if (l1.y < r2.y || l2.y < r1.y)
        return false;
    return true;
}
```
14. Convert Sorted Singly Linked List to Balanced Binary Search Tree (BST) [note own solution from leetcode here]
O(N) space, O(N log N) time
```
public TreeNode sortedListToBST(ListNode head) {
    if(head == null)
        return null;
    ListNode fast = head;
    ListNode slow = head;
    ListNode prev =null; 
    while(fast != null && fast.next != null)
    {
        fast = fast.next.next;
        prev =slow;
        slow=slow.next;
    }
    TreeNode root = new TreeNode(slow.val);
    if(prev != null)
        prev.next = null;
    else
        head  = null;
        
    root.left = sortedListToBST(head);
    root.right = sortedListToBST(slow.next);
    return root;}

O(1) space(not including space for creating the tree or recursion stack space) and O(N) time:

private ListNode node;

public TreeNode sortedListToBST(ListNode head) {
        if(head == null){
                return null;
        }
        
        int size = 0;
        ListNode runner = head;
        node = head;
        
        while(runner != null){
                runner = runner.next;
                size ++;
        }
        
        return inorderHelper(0, size - 1);}

public TreeNode inorderHelper(int start, int end){
        if(start > end){
                return null;
        }
        
        int mid = start + (end - start) / 2;
        TreeNode left = inorderHelper(start, mid - 1);
        
        TreeNode treenode = new TreeNode(node.val);
        treenode.left = left;
        node = node.next;

        TreeNode right = inorderHelper(mid + 1, end);
        treenode.right = right;
        
        return treenode;}
```
Convert a sorted Doubly Linked List to Balanced Binary Search Tree[O]
Given a doubly linked list in sorted order with previous and next nodes. Convert the doubly linked list to a binary search tree with left as previous node and right as next node.
Consider the list below:

The list should be converted to following BST:

We recursively traverse to the leaves and then create the tree upwards from the leaves to the root.
Step 1. Calculate the length of the linked list.
Step 2. Recursively create left sub tree from first half nodes.
Step 3. Make middle node as the root and node returned from previous call (Step 2) as left child of the root.
Step 4. Move head to next node.
Step 5. Recursively create the right sub tree from second half nodes.
Step 6. Return root.
```
private ListNode convertDllToBST(int len) {
        if (len == 0) {
            return null;
        }

        ListNode left = convertDllToBST(len / 2);
        ListNode root = head;
        root.prev = left;
        head = head.next;
        ListNode right = convertDllToBST(len - (len / 2) - 1);
        root.next = right;
        return root;
    }
```
15. Clone graph
```
 # Using BFS
from collections import deque
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: UndirectedGraphNode
        :rtype: UndirectedGraphNode
        """
        if node is None:
            return None
        q = deque([node])
        map = {}
        root_copy = UndirectedGraphNode(node.label)
        map[node] = root_copy
        while q:
            node = q.popleft()
            for neighbor in node.neighbors:
                if neighbor not in map:
                    p = UndirectedGraphNode(neighbor.label)
                    q.append(neighbor)
                    map[node].neighbors.append(p)
                    map[neighbor] = p
                else:
                    map[node].neighbors.append(map[neighbor])

        return root_copy

 Using DFS # Definition for a undirected graph node
# class UndirectedGraphNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution(object):
    def clone_rec(self, root, map):
        if root is None:
            return None
        root_copy = UndirectedGraphNode(root.label)
        map[root] = root_copy

        for neighbor in root.neighbors:
            if neighbor not in map:
                root_copy.neighbors.append(self.clone_rec(neighbor, map))
            else:
                root_copy.neighbors.append(map[neighbor])
        return root_copy

    def cloneGraph(self, node):
        """
        :type node: UndirectedGraphNode
        :rtype: UndirectedGraphNode
        """
        map = {}
        return self.clone_rec(node, map)
```

17.  Construct Binary Tree From Inorder and Preorder/Postorder Traversal [Ask interviewer about duplicates in tree ]
Caveats:
 1. Worst case is O(n^2) for any such algorithm for tree construction[left/right skeweed]
2. Equal numbers: impossible to construct tree
3. preorder.pop(0) is ok since ignorer.index anyways takes linear time.

Recommended approach
```java
public TreeNode buildTree(int[] preorder, int[] inorder) {
    return helper(0, 0, inorder.length - 1, preorder, inorder);
}

public TreeNode helper(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
    if (preStart > preorder.length - 1 || inStart > inEnd) {
        return null;
    }
    TreeNode root = new TreeNode(preorder[preStart]);
    int inIndex = 0; // Index of current root in inorder
    for (int i = inStart; i <= inEnd; i++) {
        if (inorder[i] == root.val) {
            inIndex = i;
        }
    }
    root.left = helper(preStart + 1, inStart, inIndex - 1, preorder, inorder);
    root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
    return root;
}
```
20. LCA of Binary Tree:
```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return None
        if root==p or root==q:
            return root
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if l is not None and r is not None:
            return root
        return l if l else r
```
21. Find number of islands . Ans
Caveat: Note that you only need to do 4 recursive calls.
Time complexity: O(ROW x COL)
```python
class Solution(object):
    def isSafe(self, row, col, grid, visited, rows, cols):
        if 0<=row<rows and 0<=col<cols and grid[row][col] == '1' and visited[row][col] is False:
            return True
        return False

    def DFS(self, row, col, grid, visited, rows, cols):
        if self.isSafe(row, col, grid, visited, rows, cols):
            visited[row][col] = True
            self.DFS(row-1, col, grid, visited, rows, cols)
            self.DFS(row+1, col, grid, visited, rows, cols)
            self.DFS(row, col-1, grid, visited, rows, cols)
            self.DFS(row, col+1, grid, visited, rows, cols)

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False for i in range(cols)] for i in range(rows)]
        count = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1' and visited[i][j] is False:
                    self.DFS(i, j, grid, visited, rows, cols)
                    count += 1
        return count
```

By union find [More about union find here: http://algs4.cs.princeton.edu/15uf/UF.java.html]
```java
public class Solution {
    class UF{
        int[] father;
        int count =0;
        public UF(char[][] grid, int m, int n) {
            father = new int[m*n];
            int k=0;
            for(int i=0;i<m;i++) {
                for(int j=0;j<n;j++) {
                    father[k] = k++;
                    if(grid[i][j] == '1') ++count;
                }
            }
        }
        void union(int node1, int node2) { // union by rank
            int root1 = find(node1);
            int root2 = find(node2);
            if(root1 == root2) return;
            if(rank[root1] < rank[root2]) father[root1] = root2;
            else if(rank[root1] > rank[root2]) father[root2] = root1;
            else {
                father[root1] = root2;
                ++rank[root1];
            }
            --count; // <---this is the key step NOTE THIS
        }
        int find(int node) {  // path compression
            while(node != father[node]) node = father[node];
            return node;
        }
    }
    int[][] distance = {{1,0},{-1,0},{0,1},{0,-1}};
    public int numIslands(char[][] grid) {  
        if (grid == null || grid.length == 0 || grid[0].length == 0)  {
            return 0;  
        }
        UF uf = new UF(grid);  
        int rows = grid.length;  
        int cols = grid[0].length;  
        for (int i = 0; i < rows; i++) {  
            for (int j = 0; j < cols; j++) {  
                if (grid[i][j] == '1') {  
                    for (int[] d : distance) {
                        int x = i + d[0];
                        int y = j + d[1];
                        if (x >= 0 && x < rows && y >= 0 && y < cols && grid[x][y] == '1') {  
                            int id1 = i*cols+j;
                            int id2 = x*cols+y;
                            uf.union(id1, id2);  
                        }  
                    }  
                }  
            }  
        }  
        return uf.count;  
    }
```
Follow up : 305. Number of Islands II
A 2d grid map of m rows and n columns is initially filled with water. We may perform an addLand operation which turns the water at position (row, col) into a land. Given a list of positions to operate, count the number of islands after each addLand operation. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example:

Given m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]].

Initially, the 2d grid grid is filled with water. (Assume 0 represents water and 1 represents land).

0 0 0 0 0 0 0 0 0

Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land.

1 0 0 0 0 0 Number of islands = 1 0 0 0

Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land.

1 1 0 0 0 0 Number of islands = 1 0 0 0

Operation #3: addLand(1, 2) turns the water at grid[1][2] into a land.

1 1 0 0 0 1 Number of islands = 2 0 0 0

Operation #4: addLand(2, 1) turns the water at grid[2][1] into a land.

1 1 0 0 0 1 Number of islands = 3 0 1 0

We return the result as an array: [1, 1, 2, 3]

Ans:  The algorithm runs in O((M+N) log* N)= O(M+N) where M is the number of operations ( unite and find ), N is the number of objects, log* is iterated logarithm while the naive runs in O(MN).

//Time O(K log * mn) where k is the length of positions array
```
public class Solution {
    int cnt= 0;
    class UF{
            int[] father;
            public UF(int m, int n) {
                father = new int[m*n];
                Arrays.fill(father, -1);
            }
            void union(int node1, int node2) { // need to use union by rank
                if(find(node1) == find(node2)) return;
                father[find(node1)] = find(node2);
                --cnt;
            }
            int find(int node) { 
                while(node != father[node]) node = father[node];
                return node;
            }
    }

    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> numIslands = new ArrayList<>();
        UF uf = new UF(m, n);
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for(int k=0;k<positions.length;k++) {
            int i = positions[k][0];
            int j = positions[k][1];            ++cnt; // important            int root = i*n+j;
            uf.father[root] = root;
            int index;
            for(int[] dir: dirs) {
                int x = i+dir[0];
                int y = j+dir[1];
                index = x*n+y;
                if(x<0 || x>=m || y<0 || y>=n || uf.father[index] == -1) continue;
                uf.union(root, index);
            }
            numIslands.add(cnt);
    
        } 
        return numIslands;
    }}
```
Description: Initially assume every cell are in non-island set {-1}. When point A is added, we create a new root, i.e., a new island. Then, check if any of its 4 neighbors belong to the same island. If not, union the neighbor by setting the root to be the same. Remember to skip non-island cells.

22. Word search

24. Longest consecutive subsequence: Given an unsorted array of integers, find the length of the longest consecutive elements
sequence. For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
Your algorithm should run in O(n) complexity.
Algorithm:

1) Create an empty hash.2) Insert all array elements to hash.3) Do following for every element arr[i]....a) Check if this element is the starting point of a        subsequence.  To check this, we simply look for      arr[i] - 1 in hash, if not found, then this is      the first element a subsequence.            If this element is a first element, then count        number of elements in the consecutive starting        with this element.      If count is more than current res, then update          res.


Solution
```
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        hash = {}
        n = len(nums)
        if n in [0,1]:
            return n
        for n in nums:
            hash[n] = 1
        ans = 0
        for key in hash.keys():
            if key-1 not in hash: # first element of sorted subsequence
                x = key
                times = 1
                while x+1 in hash:
                    times += 1
                    x = x+1
                if ans < times:
                    ans = times
        return ans
```
25. Longest palindromic substring
Solution
```
class Solution:
    def get_palindrome(self, str,i,j):
        n = len(str)
        while i >=0 and j < n and str[i] == str[j]:
            i -= 1
            j += 1
        return str[i+1:j]

    # @param {string} s
    # @return {string}
    def longestPalindrome(self, s):
        ans = ''
        n = len(s)
        if n in [0, 1]:
            return s
        for k in range(0,n):
            odd_palin = self.get_palindrome(s, k, k)
            if len(odd_palin) > len(ans):
                ans = odd_palin
            even_palin = self.get_palindrome(s, k, k+1)
            if len(even_palin) > len(ans):
                ans = even_palin
        return ans
```
By Rolling hash technique: [https://discuss.leetcode.com/topic/41599/8-line-o-n-method-using-rabin-karp-rolling-hash/3]
```
//got it fully[awesome algo]
public class Solution {
    public String shortestPalindrome(String s) {
        long B=29, mod=1000_000_007, pow=1;
        int pos=-1; // -1 so that it would work for empty strings too.
        long hash1=0, hash2=0;
        for(int i=0;i<s.length();i++, pow = pow * B % mod) {
            int val = s.charAt(i) - 'a' + 1;
            hash1 = (hash1 * B + val) % mod;
            hash2 = (hash2 + val * pow) % mod;
            if(hash1 == hash2) pos = i;
        }
        return new StringBuilder().append(s.substring(pos+1)).reverse().append(s).toString();
    }
}
```
Explanation:

Consider a decimal example. Say we are given a number 7134. If we read it from left to right, we get 7134. And 4317 if we read it from right to left.

hash1 is the left--to-right fashion:

- hash1 = 0
- hash1 = 0 * 10 + 7 = 7
- hash1 = 7 * 10 + 1 = 71
- hash1 = 71 * 10 + 3 = 713
- hash1 = 713 * 10 + 4 = 7134

hash2 is the right-to-left fashion:

- hash2 = 0
- hash2 = 0 + 7 * 1 = 7
- hash2 = 7 + 1 * 10 = 17
- hash2 = 17 + 3 * 100 = 317
- hash2 = 317 + 4 * 1000 = 4317

A palindrome must be read the same from left to right and from right to left. So in this case, 7134 is not a palindrome.

Above is an example for the decimal case, and for rolling hashing, the only differences are:

- Base is not 10, but any constant >= 26.
- hash1 and hash2 are not the exact value, but the exact value modulo a big prime. (Since the exact value is too large to fit in a 32-bit integer.)

More rolling hash problems: http://www.infoarena.ro/blog/rolling-hash

Dynamic Programmig Approach:
```
public int longestPalindromeDynamic(char []str){
        boolean T[][] = new boolean[str.length][str.length];
        
        for(int i=0; i < T.length; i++){
            T[i][i] = true;
        }
        
        int max = 1;
        for(int l = 2; l <= str.length; l++){
            int len = 0;
            for(int i=0; i < str.length-l+1; i++){
                int j = i + l-1;
                len = 0;
                if(l == 2){
                    if(str[i] == str[j]){
                        T[i][j] = true;
                        len = 2;
                    }
                }else{
                    if(str[i] == str[j] && T[i+1][j-1]){
                        T[i][j] = true;
                        len = j -i + 1;
                    }
                }
                if(len > max){
                    max = len;
                }
            }
        }
        return max;}
```
26. Permutation of a list

Permutations : https://leetcode.com/problems/permutations/
```
public List<List<Integer>> permute(int[] nums) {
  List<List<Integer>> list = new ArrayList<>();
  // Arrays.sort(nums); // not necessary
  backtrack(list, new ArrayList<>(), nums);
  return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums){
  if(tempList.size() == nums.length){
      list.add(new ArrayList<>(tempList));
  } else{
      for(int i = 0; i < nums.length; i++){
        if(tempList.contains(nums[i])) continue; // element already exists, skip
        tempList.add(nums[i]);
        backtrack(list, tempList, nums);
        tempList.remove(tempList.size() - 1);
      }
  }
}

Permutations II (contains duplicates) : https://leetcode.com/problems/permutations-ii/

public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, new boolean[nums.length]);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, boolean [] used){
    if(tempList.size() == nums.length){
        list.add(new ArrayList<>(tempList));
    } else{
        for(int i = 0; i < nums.length; i++){
            if(used[i] || i > 0 && nums[i] == nums[i-1] && !used[i - 1]) continue;
            used[i] = true;
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, used);
            used[i] = false;
            tempList.remove(tempList.size() - 1);
        }
    }
}

Another way to do permutation

    public static void printPermutations(Object[] array) {
        printPermutations(array, 0);
    }

    private static void printPermutations(Object[] array, int i) {
        if (i == array.length) {
            for (Object obj : array) System.out.print(" " + obj);
            System.out.println();
            return;
        }

        for (int j = i; j < array.length; ++j) {
            swap(array, i, j);
            printPermutations(array, i + 1);
            swap(array, i, j);
        }
    }

    private static void swap(Object[] array, int i, int j) {
        Object tmp = array[j];
        array[j] = array[i];
        array[i] = tmp;
    }
```
Next permutation
Algorithm from wiki

The following algorithm generates the next permutation lexicographically after a given permutation. It changes the given permutation in-place.

- Find the largest index k such that a[k] < a[k + 1]. If no such index exists, the permutation is the last permutation.
- Find the largest index l greater than k such that a[k] < a[l].
- Swap the value of a[k] with that of a[l].
- Reverse the sequence from a[k + 1] up to and including the final element a[n].

For example, given the sequence [1, 2, 3, 4] which starts in a weakly increasing order, and given that the index is zero-based, the steps are as follows:

- Index k = 2, because 3 is placed at an index that satisfies condition of being the largest index that is still less than a[k + 1] which is 4.
- Index l = 3, because 4 is the only value in the sequence that is greater than 3 in order to satisfy the condition a[k] < a[l].
- The values of a[2] and a[3] are swapped to form the new sequence [1,2,4,3].
- The sequence after k-index a[2] to the final element is reversed. Because only one value lies after this index (the 3), the sequence remains unchanged in this instance. Thus the lexicographic successor of the initial state is permuted: [1,2,4,3].

Following this algorithm, the next lexicographic permutation will be [1,3,2,4], and the 24th permutation will be [4,3,2,1] at which point a[k] <a[k + 1] does not exist, indicating that this is the last permutation.
```
 class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        for i in range(n-2, -1, -1):
            if nums[i] < nums[i+1]:
                for j in range(n-1, i, -1):
                    if nums[i] < nums[j]:
                        nums[i], nums[j] = nums[j], nums[i]
                        if i+1 < n:
                            start = i+1
                            end = n-1
                            while start < end:
                                nums[start] , nums[end] = nums[end], nums[start]
                                start += 1
                                end -= 1
                            return
        nums.sort()
```
Palindrome partitioning:
Given a string s, partition s such that every substring of the partition is a palindrome.
Return all possible palindrome partitioning of s.
For example, given s = "aab",
Return
[
  ["aa","b"],
  ["a","a","b"]
]
Palindrome Partitioning : https://leetcode.com/problems/palindrome-partitioning/
```
public List<List<String>> partition(String s) {
  List<List<String>> list = new ArrayList<>();
  backtrack(list, new ArrayList<>(), s, 0);
  return list;
}

public void backtrack(List<List<String>> list, List<String> tempList, String s, int start){
  if(start == s.length())
      list.add(new ArrayList<>(tempList));
  else {
      for(int i = start; i < s.length(); i++) {
        if(isPalindrome(s, start, i)){
          tempList.add(s.substring(start, i + 1));
            backtrack(list, tempList, s, i + 1);
            tempList.remove(tempList.size() - 1);
        }
      }
  }
}

public boolean isPalindrome(String s, int low, int high){
  while(low < high)
      if(s.charAt(low++) != s.charAt(high--)) return false;
  return true;
}
```
28. Permutations II [with duplicates]

Check above

29. Generate balanced parentheses
```
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        s = ""
        ans = []
        self.generateParenthesisRec(n, 0, 0, s, ans)
        return ans

    def generateParenthesisRec(self, n, left, right, str, ans):
        if left >= n and right >= n:
            ans.append(str)
            return

        if left < n:
            str += '('
            self.generateParenthesisRec(n, left+1, right, str, ans)
            str = str[:-1]
        if right < left:
            str += ')'
            self.generateParenthesisRec(n, left, right+1, str, ans)
            str = str[:-1]
```
30. Combination: Picking 'k' items from a list of 'n' - Recursion
```
public class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> list = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        backtrack(list, temp, n, k, 1);
        return list;
    }
    public void backtrack(List<List<Integer>> list, List<Integer> temp, int n, int k, int start){
        if(temp.size() == k) { list.add(new ArrayList<>(temp)); return; }
        for(int i=start;i<=n;i++){
            temp.add(i);
            backtrack(list, temp, n, k, i+1);
            temp.remove(temp.size()-1);
        }
    }
}
```
Power set
-===
Subsets : https://leetcode.com/problems/subsets/
```
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, 0);
    return list;
}

private void backtrack(List<List<Integer>> list , List<Integer> tempList, int [] nums, int start){
    list.add(new ArrayList<>(tempList));
    for(int i = start; i < nums.length; i++) {
        tempList.add(nums[i]);
        backtrack(list, tempList, nums, i + 1);
        tempList.remove(tempList.size() - 1);
    }
}
```
Subsets II (contains duplicates) : https://leetcode.com/problems/subsets-ii/
```
public List<List<Integer>> subsetsWithDup(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, 0);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, int start){
    list.add(new ArrayList<>(tempList));
    for(int i = start; i < nums.length; i++){
        if(i > start && nums[i] == nums[i-1]) continue; // skip duplicates
        tempList.add(nums[i]);
        backtrack(list, tempList, nums, i + 1);
        tempList.remove(tempList.size() - 1);
    }
}
```
Combination Sum: Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T. The same repeated number may be chosen from C unlimited number of times.

Combination Sum : https://leetcode.com/problems/combination-sum/
```
public List<List<Integer>> combinationSum(int[] nums, int target) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, target, 0);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, int remain, int start){
    if(remain < 0) return;
    else if(remain == 0) list.add(new ArrayList<>(tempList));
    else{
        for(int i = start; i < nums.length; i++){
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, remain - nums[i], i); // not i + 1 because we can reuse same elements
            tempList.remove(tempList.size() - 1);
        }
    }
}
```
Combination Sum II (can't reuse same element) : https://leetcode.com/problems/combination-sum-ii/
```
public List<List<Integer>> combinationSum2(int[] nums, int target) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, target, 0);
    return list;

}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, int remain, int start){
    if(remain < 0) return;
    else if(remain == 0) list.add(new ArrayList<>(tempList));
    else{
        for(int i = start; i < nums.length; i++){
            if(i > start && nums[i] == nums[i-1]) continue; // skip duplicates
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, remain - nums[i], i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }
}
```
31. n Queen-II
```
def totalNQueens(self, n):
    self.res = 0
    self.dfs([-1]*n, 0)
    return self.res

def dfs(self, board, index):
    if index == len(board):
        self.res += 1
        return
    for i in xrange(len(board)):
        if self.check(board, index, i):
            board[index] = i
            self.dfs(board, index+1)

# check whether kth queen can be placed
# in column j
def check(self, board, k, j):
    for i in xrange(k):
        if board[i] == j or k-i == abs(j-board[i]):
            return False
    return True
```
33. Finding all upper/lower case combinations of a word
```
public static void comb4(String word) {
    comb4(word,new char[word.length()],0);
}

private static void comb4(String word, char[] accu, int index) {
    if(index == word.length()) {
        System.out.println(accu);
    } else {
        char ch = word.charAt(index);
        accu[index] = Character.toLowerCase(ch);
        comb4(word, accu , index+1);
        accu[index] = Character.toUpperCase(ch);
        comb4(word, accu, index+1);
    }
}
```
35. Check if a binary tree is  a subtree of another tree
```
/* A utility function to check whether trees with roots as root1 and
  root2 are identical or not */
bool areIdentical(struct node * root1, struct node *root2)
{
    /* base cases */
    if (root1 == NULL && root2 == NULL)
        return true;

    if (root1 == NULL || root2 == NULL)
        return false;

    /* Check if the data of both roots is same and data of left and right
      subtrees are also same */
    return (root1->data == root2->data  &&
            areIdentical(root1->left, root2->left) &&
            areIdentical(root1->right, root2->right) );
}

/* This function returns true if S is a subtree of T, otherwise false */
bool isSubtree(struct node *T, struct node *S)
{
    /* base cases */
    if (S == NULL)
        return true;

    if (T == NULL)
        return false;

    /* Check the tree with root as current node */
    if (areIdentical(T, S))
        return true;

    /* If the tree with root as current node doesn't match then
      try left and right subtrees one by one */
    return isSubtree(T->left, S) ||
          isSubtree(T->right, S);
}
```
36. Dutch national flag problem.

The problem was posed with three colours, here `0′, `1′ and `2′. The array is divided into four sections:

- a[1..Lo-1] zeroes (red)
- a[Lo..Mid-] ones (white)
- a[Mid..Hi] unknown
- a[Hi+1..N] twos (blue)

The unknown region is shrunk while maintaining these conditions

- Lo := 1; Mid := 1; Hi := N;
- while Mid <= Hi do
    - Invariant: a[1..Lo-1]=0 and a[Lo..Mid-1]=1 and a[Hi+1..N]=2; a[Mid..Hi] are unknown.
    - case a[Mid] in
        - 0: swap a[Lo] and a[Mid]; Lo++; Mid++
        - 1: Mid++
        - 2: swap a[Mid] and a[Hi]; Hi–
```
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        lo = 0
        mid = 0
        hi = len(nums) - 1
        while mid <= hi:
            if nums[mid] == 0:
                nums[lo], nums[mid] = nums[mid], nums[lo]
                lo += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            elif nums[mid] == 2:
                nums[mid], nums[hi] = nums[hi], nums[mid]
                hi -= 1
```

37. Serialize and de-serialize a binary tree:
```
Java:
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }
    StringBuilder buildString(TreeNode root, StringBuilder sb){
        if(root == null){
            sb.append("# ");
        } else {
            sb.append(root.val+" ");
            buildString(root.left, sb);
            buildString(root.right, sb);
        }
        return sb;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        return deserializeRec(new Scanner(data));
    }
    public TreeNode deserializeRec(Scanner sc){ // can use StringTokenizer too instead of Scanner
        if(!sc.hasNext()) return null;
        String s = sc.next();
        if(s.equals("#")) return null;
        TreeNode t = new TreeNode(Integer.valueOf(s));
        t.left = deserializeRec(sc);
        t.right = deserializeRec(sc);
        return t;
    }
```
38. Serialize and de-serialize an n-ary tree

Ans:
The idea is the following:
The idea is to store an ‘end of children’ marker with every node. The following diagram shows serialization where ‘)’ is used as end of children marker. The diagram is taken from here.

Code:
```
// This function stores the given N-ary tree in a file pointed by fp
void serialize(Node *root, FILE *fp)
{
    // Base case
    if (root == NULL) return;

    // Else, store current node and recur for its children
    fprintf(fp, "%c ", root->key);
    for (int i = 0; i < N && root->child[i]; i++)
        serialize(root->child[i],  fp);

    // Store marker at the end of children
    fprintf(fp, "%c ", MARKER);
}

// This function constructs N-ary tree from a file pointed by 'fp'.
// This functionr returns 0 to indicate that the next item is a valid
// tree key. Else returns 0
int deSerialize(Node *&root, FILE *fp)
{
    // Read next item from file. If theere are no more items or next
    // item is marker, then return 1 to indicate same
    char val;
    if ( !fscanf(fp, "%c ", &val) || val == MARKER )
      return 1;

    // Else create node with this item and recur for children
    root = newNode(val);
    for (int i = 0; i < N; i++)
      if (deSerialize(root->child[i], fp))
        break;
    // Finally return 0 for successful finish
    return 0;
}
```
39. n-ary tree to binary tree conversion and vice versa

Ans:

Algorithm:
1. Link the child nodes for a parent node that are at the same level.
2. Then from the links of the original tree, the link from the parent to the first (or the leftmost) child is preserved and the subsequent
links to the children are discarded.
3. Then keeping the root as the center, the tree is rotated by 45º clockwise. The tree obtained is the desired binary tree.

40. Find prime factors
```
def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
      primfac.append(n)
    return primfac
```
41.Longest Substring Without Repeating Characters
```
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        ans = 0
        start = 0
        hash = {}
        for i in range(n):
            if s[i] in hash:
                start = max(start, hash[s[i]]+1)
            hash[s[i]] = i
            ans = max(ans, i-start+1)
        return ans
```
42. Create a tree type and determine if a given tree is unival.
* A tree is unival if for all nodes in the tree, they have the same value. Given a tree, determine the number of sub-trees which are unival.

Ans:
Notes: Finding number of unival subtrees in O(n) time is a good question. Brute force is quadratic
For linear time: Do post order traversal from bottom up and if any of the sub-trees is not uni-val, return only leftUnivalCount + rightUnivalCount otherwise return leftUnivalCount+rightUnivalCount+1. You can use static variable : `isSubTreeUnival` to see if you ever got a non-unival subtree while going bottom up. This will do it in linear time.
For ex, for the following tree, number of uni-val trees is 2.
      1
    /    \
 1        3

```
//O(N)
    static boolean isSubTreeUnival = true;
    static int findSingleValueTrees(Node n) {
        if(n == null) return 0;
        if(n.left == null && n.right == null) return 1;
        int count = 0;
        int lCount = findSingleValueTrees(n.left);
        int rCount = findSingleValueTrees(n.right);
        if((n.left != null && n.val != n.left.val) || (n.right != null && n.val != n.right.val))
            isSubTreeUnival = false;
        if(isSubTreeUnival == true)
            count += lCount+rCount+1;
        else
            count += lCount+rCount;
        return count;
    }

//brute force: O(n^2) time
 bool isSameValue(Node* n, int value) {
    if (n == NULL) return true;
    if (n->value != value) return false;
    return isSameValue(root->left, value) && isSameValue(root->right, value);
}

bool isUnival(Node* root) {
    if (root == NULL) return true;
    return isSameValue(root->left, root->value) &&
           isSameValue(root->right, root->value)
}
int univalSubTrees(Node* n) {
    if (isUnival(n)) {
        return 1;
    } else {
        return univalSubTree(n->left) + univalsubTree(n->right);
    }
}
```
43. Given 4 vertices, find if they form a square
Ans: Find all distances b/w various vertices and sort them : just verify if the 1st 4 are equal and next 2 are also equal

44. Trie implementation
```
class TrieNode {
    public boolean isWord;
    public TrieNode[] children = new TrieNode[26]; //or Map<Character, TrieNode> children; // either is fine
    public TrieNode() {}
}

public class Trie {
    private TrieNode root;
    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        TrieNode ws = root;
        for(int i = 0; i < word.length(); i++){
            char c = word.charAt(i);
            if(ws.children[c - 'a'] == null){
                ws.children[c - 'a'] = new TrieNode();
            }
            ws = ws.children[c - 'a'];
        }
        ws.isWord = true;
    }

    public boolean search(String word) {
        TrieNode ws = root;
        for(int i = 0; i < word.length(); i++){
            char c = word.charAt(i);
            if(ws.children[c - 'a'] == null) return false;
            ws = ws.children[c - 'a'];
        }
        return ws.isWord;
    }

    public boolean startsWith(String prefix) {
        TrieNode ws = root;
        for(int i = 0; i < prefix.length(); i++){
            char c = prefix.charAt(i);
            if(ws.children[c - 'a'] == null) return false;
            ws = ws.children[c - 'a'];
        }
        return true;
    }
}
```
45. How do you find if a number is a power of 3 in less than O(n) time ?

Ans:
Cover the number into base 3. Only the first digit should be 1 and rest should be all 0s.
```java
public class Solution {
    public boolean isPowerOfThree(int n) {
        return Integer.toString(n, 3).matches("^10*$");
    }
}
```
```cpp
public boolean isPowerOfThree(int n) {
    if(n>1)
        while(n%3==0) n /= 3;
    return n==1;
}
```
Follow up: Find if a number is a power of 4 without loops

```java
public boolean isPowerOfFour(int num){return Integer.toString(num,4).matches("10*"); }

//Another way
bool isPowerOfFour(int num) {
    return num > 0 && (num & (num - 1)) == 0 && (num - 1) % 3 == 0;
}
```
46. BST TO DLL[as well as circular DLL]

https://leetcode.com/discuss/77722/8-lines-of-python-solution-post-order-traversal
// only for converting to a singly linked list, also nodes are in the pre-order traversal of the BST.
public void flatten(TreeNode root) {
    flatten(root,null);
}
private TreeNode flatten(TreeNode root, TreeNode pre) {
    if(root==null) return pre; // this line is very important
    pre=flatten(root.right,pre);
    pre=flatten(root.left,pre);
    root.right=pre;
    root.left=null;
    pre=root;
    return pre;
}

From: CTCI

class TreeList {
    /*    helper function -- given two list nodes, join them    together so the second immediately follow the first.    Sets the .next of the first and the .previous of the second.    */
    public static void join(Node a, Node b) {
        a.right = b;
        b.left = a;
    }

    
    /*    helper function -- given two circular doubly linked    lists, append them and return the new list.    */
    public static Node append(Node a, Node b) {
        // if either is null, return the other
        if (a==null) return(b);
        if (b==null) return(a);
        
        // find the last node in each using the .previous pointer
        Node part1 = a.left;
        Node part3 = b.left;
        
        // join the two together to make it connected and circular
        join(part1, b);
        join(part3, a);
        
        return(a);
    }

    
    /*    --Recursion--    Given an ordered binary tree, recursively change it into    a circular doubly linked list which is returned.    */
    public static Node treeToList(Node root) {
        // base case: empty tree -> empty list
        if (root==null) return(null);
        
        // Recursively do the subtrees (leap of faith!)
        Node part1 = treeToList(root.left);
        Node part3 = treeToList(root.right);
        
        // Make the single root node into a list length-1
        // in preparation for the appending
        root.left = root;
        root.right = root;
        
        // At this point we have three lists, and it's
        // just a matter of appending them together
        // in the right order (part1, root, part3)
        part1 = append(part1, root);
        part1 = append(part1, part3);
        
        return(part1);
    }}
```
"""2. Given an unsorted of array of integers, and an integer k. Return the top k most frequent elements from the array.
input : [2, 3, 2, 4, 2, 3, 2]
k : 2
=> [2, 3]
k : 1 => [2]
"""
```
def find_top_k_elements(input, k):
    hash = {}
    for i in input:
        if i in hash:
            hash[i] += 1
        else:
            hash[i] = 1

    heap = []
    for key, val in hash.iteritems():
        n = len(heap)
        if n < k:
            heapq.heappush(heap, (val, key))
        else:
            top = heap[0]
            if top[0] < val:
                heapq.heappop(heap)
                heapq.heappush(heap, (val, key))
    ans = []
    for item in heap:
        ans.append(item[1])
    return ans
```
# O(nlogk)
# How to handle when there is an infinite stream of sorted numbers ? And you get a query for k most frequent numbers
# k is given, sorted strem
# 2 2 3 3 4 ? 4 4 ? 4 ? 5 5 5 ..........................

Approach from EPI: problem: 15.8, slightly different from the above problem, the numbers are coming in a stream now and after each number comes, you need to find out the K most frequent integers: Use a balanced BST(TreeMap[red-black tree] in java?) + hash map to achieve O(k+log m) time solution where m is the number of distinct integers visited so far.

48. Spiral Matrix: Just remember the highlighted conditions.
```
# Time:  O(m * n)
# Space: O(1)
#
# Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
#
# For example,
# Given the following matrix:
#
# [
#  [ 1, 2, 3 ],
#  [ 4, 5, 6 ],
#  [ 7, 8, 9 ]
# ]
# You should return [1,2,3,6,9,8,7,4,5].
#

class Solution:
    # @param matrix, a list of lists of integers
    # @return a list of integers
    def spiralOrder(self, matrix):
        result = []
        if matrix == []:
            return result

        left, right, top, bottom = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
`
        while left <= right and top <= bottom:
            for j in xrange(left, right + 1):
                result.append(matrix[top][j])
            for i in xrange(top + 1, bottom):
                result.append(matrix[i][right])
            for j in reversed(xrange(left, right + 1)):
                if top < bottom:
                    result.append(matrix[bottom][j])
            for i in reversed(xrange(top + 1, bottom)):
                if left < right:
                    result.append(matrix[i][left])
            left, right, top, bottom = left + 1, right - 1, top + 1, bottom - 1

        return result
```
49. Jump game ii:

DP: O(N^2) : similar to LIS DP
Greedy: O(N)
http://www.geeksforgeeks.org/minimum-number-of-jumps-to-reach-end-of-a-given-array/

Given an array of integers where each element represents the max number of steps that can be made forward from that element. Write a function to return the
minimum number of jumps to reach the end of the array (starting from the first element). If an element is 0, then cannot move through that element.

Example:

Input: arr[] = {1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9}
Output: 3 (1-> 3 -> 8 ->9)

Method 2 (Dynamic Programming)  : Similar to LIS DP // TLE on leetcode.
In this method, we build a jumps[] array from left to right such that jumps[i] indicates the minimum number of jumps needed to reach arr[i] from arr[0]. Finally, we return jumps[n-1].
```
#include <stdio.h>
#include <limits.h>


int min(int x, int y) { return (x < y)? x: y; }


// Returns minimum number of jumps to reach arr[n-1] from arr[0]
int minJumps(int arr[], int n)
{
    int *jumps = new int[n]; // jumps[n-1] will hold the result
    int i, j;


    if (n == 0 || arr[0] == 0)
        return INT_MAX;


    jumps[0] = 0;


    // Find the minimum number of jumps to reach arr[i]
    // from arr[0], and assign this value to jumps[i]
    for (i = 1; i < n; i++)
    {
        jumps[i] = INT_MAX;
        for (j = 0; j < i; j++)
        {
            if (i <= j + arr[j] && jumps[j] != INT_MAX)
            {
                jumps[i] = min(jumps[i], jumps[j] + 1);
                break;
            }
        }
    }
    return jumps[n-1];
}


  //greedy simplest
// step is the number of steps left
// increment jump only when you run out of steps. next step is maxReach- i because that is the maximum we can reach in the next jump
//http://www.lifeincode.net/programming/leetcode-jump-game-and-jump-game-ii-java/

public class Solution {
    public int jump(int[] A) {
        if (A.length <= 1)
            return 0;
        int maxReach = A[0];
        int step = A[0];
        int jump = 1;
        for (int i = 1; i < A.length; i++) {
            if (i == A.length - 1)
                return jump;
            if (i + A[i] > maxReach)
                maxReach = i + A[i];
            step--;
            if (step == 0) {
                jump++;
                step = maxReach - i;
            }
        }
        return jump;
    }
}
```
50. Count number of trees
```
For the key values 1...numKeys, how many structurally unique
binary search trees are possible that store those keys.
Strategy: consider that each value could be the root.
Recursively find the size of the left and right subtrees.
*/
int countTrees(int numKeys) {
if (numKeys <=1) {
  return(1);
  }
else {
  // there will be one value at the root, with whatever remains
  // on the left and right each forming their own subtrees.
  // Iterate through all the values that could be the root...
  int sum = 0;
  int left, right, root;
  for (root=1; root<=numKeys; root++) {
      left = countTrees(root - 1); // very important here
      right = countTrees(numKeys - root);
      // number of possible trees with this root == left*right
      sum += left*right;
  }
  return(sum);
  }
}

Generate all such trees:

public static List<BinaryTreeNode<Integer>> generateAllBinaryTrees(
      int numNodes) {
    List<BinaryTreeNode<Integer>> result = new ArrayList<>();
    if (numNodes == 0) { // Empty tree, add as an null.
      result.add(null);
    }

    for (int numLeftTreeNodes = 0; numLeftTreeNodes < numNodes;
        ++numLeftTreeNodes) {
      int numRightTreeNodes = numNodes - 1 - numLeftTreeNodes;
      List<BinaryTreeNode<Integer>> leftSubtrees
          = generateAllBinaryTrees(numLeftTreeNodes);
      List<BinaryTreeNode<Integer>> rightSubtrees
          = generateAllBinaryTrees(numNodes - 1 - numLeftTreeNodes);
      // Generates all combinations of leftSubtrees and rightSubtrees.
      for (BinaryTreeNode<Integer> left : leftSubtrees) {
        for (BinaryTreeNode<Integer> right : rightSubtrees) {
          result.add(new BinaryTreeNode<>(0, left, right));
        }
      }
    }
    return result;
  }
```
51. Find median of 2 sorted arrays

Java solution [https://discuss.leetcode.com/topic/28602/concise-java-solution-based-on-binary-search]
=========

The key point of this problem is to ignore half part of A and B each step recursively by comparing the median of remaining A and B:

if (aMid < bMid) Keep [aRight + bLeft]else Keep [bRight + aLeft]

As the following: time=O(log(m + n))
```
public double findMedianSortedArrays(int[] A, int[] B) {
            int m = A.length, n = B.length;
            int l = (m + n + 1) / 2;
            int r = (m + n + 2) / 2;
            return (getkth(A, 0, B, 0, l) + getkth(A, 0, B, 0, r)) / 2.0;
        }

public double getkth(int[] A, int aStart, int[] B, int bStart, int k) { // finds k-th smallest element overall
        if (aStart > A.length - 1) return B[bStart + k - 1];
        if (bStart > B.length - 1) return A[aStart + k - 1];
        if (k == 1) return Math.min(A[aStart], B[bStart]);

        int aMid = Integer.MAX_VALUE, bMid = Integer.MAX_VALUE;
        if (aStart + k/2 - 1 < A.length) aMid = A[aStart + k/2 - 1];
        if (bStart + k/2 - 1 < B.length) bMid = B[bStart + k/2 - 1];

        if (aMid < bMid)
            return getkth(A, aStart + k/2, B, bStart,       k - k/2);// Check: aRight + bLeft
        else
            return getkth(A, aStart,       B, bStart + k/2, k - k/2);// Check: bRight + aLeft
}

 class Solution:
    # @param {integer[]} nums1
    # @param {integer[]} nums2
    # @return {float}
    def findMedianSortedArrays(self, nums1, nums2):
        m = len(nums1) + len(nums2)

        if m % 2 == 1:
            return self.kth(nums1, nums2, m / 2)
        else:
            return float(self.kth(nums1, nums2, m / 2) + self.kth(nums1, nums2, m / 2 - 1)) / 2

    def kth(self, a, b, k):
        if not a:
            return b[k]
        if not b:
            return a[k]

        midA, midB = len(a) / 2, len(b) / 2

        if midA + midB < k:
            if a[midA] > b[midB]:
                return self.kth(a, b[midB + 1:], k - midB - 1)
            else:
                return self.kth(a[midA + 1:], b, k - midA - 1)
        else:
            if a[midA] > b[midB]:
                return self.kth(a[:midA], b, k)
            else:
                return self.kth(a, b[:midB], k)
```
52. Interleaving String problem

Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
For example,
Given:
s1 = "aabcc",
s2 = "dbbca",
When s3 = "aadbbcbcac", return true.
When s3 = "aadbbbaccc", return false.

Ans:

DP table represents if s3 is interleaving at (i+j)th position when s1 is at ith position, and s2 is at jth position. 0th position means empty string.
So if both s1 and s2 is currently empty, s3 is empty too, and it is considered interleaving. If only s1 is empty, then if previous s2 position is
interleaving and current s2 position char is equal to s3 current position char, it is considered interleaving. similar idea applies to when s2 is empty.
when both s1 and s2 is not empty, then if we arrive i, j from i-1, j, then if i-1,j is already interleaving and i and current s3 position equal, it s interleaving.
If we arrive i,j from i, j-1, then if i, j-1 is already interleaving and j and current s3 position equal. it is interleaving.
When no repeated chars in any strings, just do like merge function of merge sort.
If repeated, call recursion and cache.
```
    public boolean isInterleaved(char str1[], char str2[], char str3[]){
        boolean T[][] = new boolean[str1.length +1][str2.length +1];
        
        if(str1.length + str2.length != str3.length){
            return false;
        }
        
        for(int i=0; i < T.length; i++){
            for(int j=0; j < T[i].length; j++){
                int l = i + j -1;
                if(i == 0 && j == 0){
                    T[i][j] = true;
                }
                else if(i == 0){
                    if(str3[l] == str2[j-1]){
                        T[i][j] = T[i][j-1];
                    }
                }
                else if(j == 0){
                    if(str1[i-1] == str3[l]){
                        T[i][j] = T[i-1][j];
                    }
                }
                else{
                    T[i][j] = (str1[i-1] == str3[l] ? T[i-1][j] : false) || (str2[j-1] == str3[l] ? T[i][j-1] : false);
                }
            }
        }
        return T[str1.length][str2.length];
    }
```
53. Longest common substring of 2 strings: s1 and s2

Ans: Easy approach of O(N^2): just slide s2 by 1 char each time[given that s2 is smaller than s1] and find the longest common
substring of s1 and s2. This has the problem that you have to slide both strings, but to alleviate that you can always call the same function
with arguments reversed.
```
def longestCommonSubstring(x, y):
    n = len(x)
    m = len(y)
    table = collections.defaultdict(int)  # a hashtable, but we'll use it as a 2D array here
    l = (0, 0)
    for i in range(n+1):     # i=0,1,...,n
        for j in range(m+1):  # j=0,1,...,m
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
                if table[i, j] > l[1]:
                    l = (i, table[i,j])
    return x[l[0]-l[1]: l[0]]
print longestCommonSubstring('axbced', 'xbcd')

This program is for Longest common subsequence problem. Substring has to be consecutive and it’s not the same as subsequence.
DP approach: O(N^2):
def longestCommonSubsequence(x, y):
    n = len(x)
    m = len(y)
    table = dict()  # a hashtable, but we'll use it as a 2D array here

    for i in range(n+1):     # i=0,1,...,n
        for j in range(m+1):  # j=0,1,...,m
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
            else:
                table[i, j] = max(table[i-1, j], table[i, j-1]) # for longest common substring problem, change this line to: table[i, j] = 0 and to get the final substring just store

    # Now, table[n, m] is the length of LCS of x and y.

    # Let's go one step further and reconstruct
    # the actual sequence from DP table:

    def recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i-1] == y[j-1]:
            return recon(i-1, j-1) + [x[i-1]]
        elif table[i-1, j] > table[i, j-1]: #index out of bounds bug here: what if the first elements in the sequences aren't equal
            return recon(i-1, j)
        else:
            return recon(i, j-1)

    return recon(n, m)

To find diff, use the same algorithm above to find LCS, then do the following:

def printDiff(C, X, Y, i, j): # C is LCS matrix of X and Y
    if i > 0 and j > 0 and X[i-1] == Y[j-1]:
        printDiff(C, X, Y, i-1, j-1)
        print "  " + X[i-1]
    else:
        if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
            printDiff(C, X, Y, i, j-1)
            print "+ " + Y[j-1]
        elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
            printDiff(C, X, Y, i-1, j)
            print "- " + X[i-1]
# See example output here: https://en.wikipedia.org/wiki/Longest_common_subsequence_problem#Print_the_diff
```
54: Longest repeated substring in a string s
Ans: Apply the same as above easy approach of O(N^2)

55. Reverse a string using recursion
```
def reverse(text):
    if len(text) <= 1:
        return text

    return reverse(text[1:]) + text[0]
```
Reverse a string word by word:

Clarification:

- What constitutes a word?
A sequence of non-space characters constitutes a word.
- Could the input string contain leading or trailing spaces?
Yes. However, your reversed string should not contain leading or trailing spaces.
- How about multiple spaces between two words?
Reduce them to a single space in the reversed string.
```
public void reverseWords(char[] s) {
    reverse(s, 0, s.length-1);  // reverse the whole string first
    int r = 0;
    while (r < s.length) {
        int l = r;
        while (r < s.length && s[r] != ' ')
            r++;
        reverse(s, l, r-1);  // reverse words one by one
        r++;
    }
}

public void reverse(char[] s, int l, int r) {
    while (l < r) {
        char tmp = s[l];
        s[l++] = s[r];
        s[r--] = tmp;
    }
}

//If string is given

public class Solution {
    public String reverseWords(String s) {
        String[] strs = s.split("\\s+”); // \\s+ : plus is important
        StringBuilder sb = new StringBuilder();
        for(int i=strs.length-1; i>=0;i--){
            sb.append(strs[i]);
            sb.append(" ");
        }
        return sb.toString().trim();
    }
}
```

56. Reverse a number
```
int reversDigits(int num)
{
    int rev_num = 0;
    while(num > 0)
    {
        rev_num = rev_num*10 + num%10;
        num = num/10;
    }
    return rev_num;
} // can also do int(str(n)[::-1]) if n is non-negative
```
57. Maximum rectangle in a histogram

Above is a histogram where width of each bar is 1, given height = [2,1,5,6,2,3].

The largest rectangle is shown in the shaded area, which has area = 10 unit.

Ans:
Following is the complete algorithm.
1) Create an empty stack.
2) Start from first bar, and do following for every bar ‘hist[i]’ where ‘i’ varies from 0 to n-1.
……a) If stack is empty or hist[i] is higher than the bar at top of stack, then push ‘i’ to stack.
……b) If this bar is smaller than the top of stack, then keep removing the top of stack while top of the stack is greater. Let the removed bar be hist[tp]. Calculate area of rectangle with hist[tp] as smallest bar. For hist[tp], the ‘left index’ is previous (previous to tp) item in stack and ‘right index’ is ‘i’ (current index).

3) If the stack is not empty, then one by one remove all bars from stack and do step 2.b for every removed bar.

```java
public int largestRectangleArea(int[] height) {
  if (height == null || height.length == 0) return 0;
  Stack<Integer> stack = new Stack<Integer>();
  int max = 0;
  int i = 0;
  while (i < height.length) {
    //push index to stack when the current height is larger than the previous one
    if (stack.isEmpty() || height[i] >= height[stack.peek()]) {
      stack.push(i);
      i++;
    } else {
    //calculate max value when the current height is less than the previous one
      int p = stack.pop();
      int h = height[p];
      int w = stack.isEmpty() ? i : i - stack.peek() - 1;
      max = Math.max(h * w, max);
    }
}
  while (!stack.isEmpty()) {
    int p = stack.pop();
    int h = height[p];
    int w = stack.isEmpty() ? i : i - stack.peek() - 1;
    max = Math.max(h * w, max);
}
return max;
}
```

58. Shuffle a linked list in O(nlogn) time.
What about the following? Perform the same procedure as merge sort. When merging, instead of selecting an element (one-by-one) from the two lists in sorted order, flip a coin. Choose whether to pick an element from the first or from the second list based on the result of the coin flip.
Algorithm.

shuffle(list):
    if list contains a single element
        return list

    list1,list2 = [],[]
    while list not empty:
        move front element from list to list1
        if list not empty: move front element from list to list2

    shuffle(list1)
    shuffle(list2)

    if length(list2) < length(list1):
        i = pick a number uniformly at random in [0..length(list2)]
        insert a dummy node into list2 at location i

    # merge
    while list1 and list2 are not empty:
        if coin flip is Heads:
            move front element from list1 to list
        else:
            move front element from list2 to list

    if list1 not empty: append list1 to list
    if list2 not empty: append list2 to list

    remove the dummy node from list

The key point for space is that splitting the list into two does not require any extra space. The only extra space we need is
to maintain log n elements on the stack during recursion. The point with the dummy node is to realize that inserting and removing
a dummy element keeps the distribution of the elements uniform.

Look at this code: https://github.com/ldamewood/leetcode/blob/master/python/median-of-two-sorted-arrays.py

59. Master Theorem
60. Suffix array: From: http://algorithmicalley.com/archive/2013/06/30/suffix-arrays.aspx

# O(n^2) implementation:
def get_suffix_array(str):
     return sorted(range(len(str)), key=lambda i: str[i:])

# n log n implementation
import time
from collections import defaultdict, Counter

def get_suffix_array(str):
    return sorted(range(len(str)), key=lambda i: str[i:])

def sort_bucket(str, bucket, order):
    d = defaultdict(list)
    for i in bucket:
        key = str[i:i+order]
        d[key].append(i)
    result = []
    for k,v in sorted(d.iteritems()):
        if len(v) > 1:
            result += sort_bucket(str, v, order*2)
        else:
            result.append(v[0])
    return result

def suffix_array_ManberMyers(str):
    return sort_bucket(str, (i for i in range(len(str))), 1)

if __name__ == "__main__":
    with open("MobyDick.txt") as f:
        m = f.read()
    str = m#[:100000]
    print len(str)
#    str = "mississipi"
    start_time = time.time()
    #x = get_suffix_array(str)
    end_time = time.time()
    print("Time for python sort was %g seconds" % (end_time - start_time))
    start_time = time.time()
    y = suffix_array_ManberMyers(str)
    end_time = time.time()
    #assert(x == y)
    print("Time for Manber Myers was %g seconds" % (end_time - start_time))

SA : Problem 1 : Given a substring P of S, locating the occurrence positions of P only requires two binary search. First search is looking for the start position in the Suffix Array and the second
search is looking for the end position in the Suffix Array.
from bisect import bisect_left,bisect_right
def search(S,A,P):
    ''' Find the left and right index boundary of P in Suffix Array A. '''
    S = S+'\0'
    suffix = [S[i:] for i in A]
    l = bisect_left(suffix, P+"\0")
    r = bisect_right(suffix,P+"\x7F")
    return (l,r)
Problem to solve: https://www.quora.com/Suffix-Arrays/Given-two-strings-s1-and-s2-What-is-the-best-algorithm-to-find-the-number-of-common-sub-strings-between-s1-and-s2-of-length-1-2-upto-min-s1-s2
Another problem: http://www.roman10.net/suffix-array-part-3-longest-common-substring-lcs/
Another resource on SA in python: http://www.cs.jhu.edu/~langmea/resources/lecture_notes/suffix_arrays.pdf
King of all resource on SAs: http://web.stanford.edu/class/cs97si/suffix-array.pdf

60. Maximum sum such that no two elements are adjacent
Question: Given an array of positive numbers, find the maximum sum of a subsequence with the constraint that no 2 numbers in the sequence should be adjacent in the array. So 3 2 7 10 should return 13 (sum of 3 and 10) or 3 2 5 10 7 should return 15 (sum of 3, 5 and 7).Answer the question in most efficient way.
Ans:
We can use Dynamic programming :
let table[i] represents maximum sum from first element to ith index element.
Then,
```
table[0]=a[0];
table[1]=max(a[0], a[1]);
for i from 2 to n-1
    table[i] = max{a[i]+table[i-2], table[i-1] };
return table[n-1];
```
61. Selection Algorithm: kth smallest in linear time
```
// using max-heap/priority-queue: O(klogn): this is k-th largest
public class Solution {
    public int findKthLargest(int[] nums, int k) {
        Queue<Integer> q = new PriorityQueue<>(nums, Collections.reverseOrder());
        for(int i=0;i<k-1;i++) q.poll();
        return q.poll();
    }
}

//copy code from somewhere else:
// Via selection algorithm
// O(N) guaranteed running time + O(1) space
public int findKthLargest(int[] nums, int k) {
        shuffle(nums);
        k = nums.length - k;
        int lo = 0;
        int hi = nums.length - 1;
        while (lo < hi) {
            final int j = partition(nums, lo, hi);
            if(j < k) {
                lo = j + 1;
            } else if (j > k) {
                hi = j - 1;
            } else {
                break;
            }
        }
        return nums[k];
    }

private void shuffle(int a[]) {
        final Random random = new Random();
        for(int ind = 1; ind < a.length; ind++) {
            final int r = random.nextInt(ind + 1);
            exch(a, ind, r);
        }
    }
```
62. Finding Top K Frequent Items

Ans: The first step is to count how many times each number appears in the file. If the file is pre-sorted then we need a single scan over the file.
Next:
Approach 1: O(N) time
Use selection algorithm to find the Kth most frequent number (on the second element of the tuple) using the Selection Algorithm in O(U) time. The Kth most frequent element partitions the array in two parts: first part containing top K most frequent elements and second part containing bottom U-K-1 frequent elements. So we get the top K most frequent elements in no particular order in O(N) time (assuming U = O(N)). They can be sorted in O(K log K) if needed. Note that although this approach runs in O(N) time, the constants hidden in the O-notation can be large. So in practice this approach can be slower than the two approaches described below.
Another implementation:
```
def partition1(arr, left, right, pivotIndex):
    arr[right], arr[pivotIndex]=arr[pivotIndex], arr[right]
    pivot=arr[right]
    swapIndex=left
    for i in range(left, right):
        if arr[i]<pivot:
            arr[i], arr[swapIndex] = arr[swapIndex], arr[i]
            swapIndex+=1
    arr[right], arr[swapIndex]=arr[swapIndex], arr[right]
    return swapIndex

def kthLargest1(arr, left, right, k):
    if not 1<=k<=len(arr):
        return
    if left==right:
        return arr[left]

    while True:
        pivotIndex=random.randint(left, right)
        pivotIndex=partition1(arr, left, right, pivotIndex)
        rank=pivotIndex-left+1
        if rank==k:
            return arr[pivotIndex]
        elif k<rank:
            return kthLargest1(arr, left, pivotIndex-1, k)
        else:
            return kthLargest1(arr, pivotIndex+1, right, k-rank)
```
Approach 2: O(N log K) time
Pick first K tuples and put them on MIN-HEAP, where a tuple (x,y) is less than a tuple (a,b) if y is less than b. The time complexity to make the min heap of size K is O(K).

Then for the remaining U - K elements, pick them one by one. If the picked element is lesser than the minimum on the heap, discard that element. Otherwise remove the min element from the head and insert the selected element in the heap. This ensures that heap contains only K elements. This delete-insert operation is O(log K) for each element.

Once we are done picking all the elements, the elements that finally remain in the min-heap are the top K frequent items which can be popped in O(K log K) time. The overall cost of this approach is O(K + (U-K) log K + K log K) = O(K + U log K). Since K < U and U = O(N), we get the time complexity of O(N log K).

Approach 3: O(N + K log N) time
This approach is similar to approach 2 but the main difference is that we make a MAX-HEAP of all the U elements. So the first step is to make the max heap of all the elements in O(U). Then remove the maximum element from the heap K times in O(K log U) time. The K removed elements are the desired most frequent elements. The time complexity of this method is O(U + K log U) and by setting U = O(N) we get O(N + K log N).

63. Topological sort
```
time
// This class represents a directed graph using adjacency// list representationclass Graph{
    private int V;  // No. of vertices
    private LinkedList<Integer> adj[]; // Adjacency List

    //Constructor
    Graph(int v)
    {
        V = v;
        adj = new LinkedList[v];
        for (int i=0; i<v; ++i)
            adj[i] = new LinkedList();
    }

    // Function to add an edge into the graph
    void addEdge(int v,int w) { adj[v].add(w); }

    // A recursive function used by topologicalSort
    void topologicalSortUtil(int v, boolean visited[],
                            Stack stack)
    {
        // Mark the current node as visited.
        visited[v] = true;
        Integer i;

        // Recur for all the vertices adjacent to this
        // vertex
        Iterator<Integer> it = adj[v].iterator();
        while (it.hasNext())
        {
            i = it.next();
            if (!visited[i])
                topologicalSortUtil(i, visited, stack);
        }

        // Push current vertex to stack which stores result
        stack.push(new Integer(v));
    }

    // The function to do Topological Sort. It uses
    // recursive topologicalSortUtil()
    void topologicalSort()
    {
        Stack stack = new Stack();

        // Mark all the vertices as not visited
        boolean visited[] = new boolean[V];
        for (int i = 0; i < V; i++)
            visited[i] = false;

        // Call the recursive helper function to store
        // Topological Sort starting from all vertices
        // one by one
        for (int i = 0; i < V; i++)
            if (visited[i] == false)
                topologicalSortUtil(i, visited, stack);

        // Print contents of stack
        while (stack.empty()==false)
            System.out.print(stack.pop() + " ");
    }

    // Driver method
    public static void main(String args[])
    {
        // Create a graph given in the above diagram
        Graph g = new Graph(6);
        g.addEdge(5, 2);
        g.addEdge(5, 0);
        g.addEdge(4, 0);
        g.addEdge(4, 1);
        g.addEdge(2, 3);
        g.addEdge(3, 1);

        System.out.println("Following is a Topological " +
                          "sort of the given graph");
        g.topologicalSort();
    }}
```
Topological sort using BFS: [from: https://www.quora.com/Can-topological-sorting-be-done-using-BFS]
Yes, topological sorting can be performed using either DFS or BFS. Either traversal order guarantees a correct topological ordering.
Some rough psuedocode (substitute queue for stack if you want DFS):
```
Some rough psuedocode (substitute queue for stack if you want DFS):
• fill(in_count, 0)
• for e in edges:
•   in_count[e.second]++
• for a in nodes:
•   if in_count[a] = 0:
•     q.push(a)
• while not q.empty():
•   cur = q.front()
•   q.pop()
•   for nxt in adj[cur]:
•     in_count[nxt]--
•     if in_count[nxt] = 0:
•       q.push(nxt)

For graph as follow:

The topological order can be:

[0, 1, 2, 3, 4, 5][0, 2, 3, 1, 5, 4]…

//Class to represent a graphclass Graph{
    int V;// No. of vertices
   
    //An Array of List which contains 
    //references to the Adjacency List of 
    //each vertex
    List <Integer> adj[];
    public Graph(int V)// Constructor
    {
        this.V = V;
        adj = new ArrayList[V];
        for(int i = 0; i < V; i++)
            adj[i]=new ArrayList<Integer>();
    }
   
    // function to add an edge to graph
    public void addEdge(int u,int v)
    {
        adj[u].add(v);
    }
    // prints a Topological Sort of the complete graph  
    public void topologicalSort()
    {
        // Create a array to store indegrees of all
        // vertices. Initialize all indegrees as 0.
        int indegree[] = new int[V];
       
        // Traverse adjacency lists to fill indegrees of
        // vertices. This step takes O(V+E) time        
        for(int i = 0; i < V; i++)
        {
            ArrayList<Integer> temp = (ArrayList<Integer>) adj[i];
            for(int node : temp)
            {
                indegree[node]++;
            }
        }
       
        // Create a queue and enqueue all vertices with
        // indegree 0
        Queue<Integer> q = new LinkedList<Integer>();
        for(int i = 0;i < V; i++)
        {
            if(indegree[i]==0)
                q.add(i);
        }
       
        // Initialize count of visited vertices
        int cnt = 0;
       
        // Create a vector to store result (A topological
        // ordering of the vertices)
        Vector <Integer> topOrder=new Vector<Integer>();
        while(!q.isEmpty())
        {
            // Extract front of queue (or perform dequeue)
            // and add it to topological order
            int u=q.poll();
            topOrder.add(u);
           
            // Iterate through all its neighbouring nodes
            // of dequeued node u and decrease their in-degree
            // by 1
            for(int node : adj[u])
            {
                // If in-degree becomes zero, add it to queue
                if(--indegree[node] == 0)
                    q.add(node);
            }
            cnt++;
        }
       
        // Check if there was a cycle     
        if(cnt != V)
        {
            System.out.println("There exists a cycle in the graph");
            return ;
        }
       
        // Print topological order          
        for(int i : topOrder)
        {
            System.out.print(i+" ");
        }
    }}// Driver program to test above functionsclass Main{
    public static void main(String args[])
    {
        // Create a graph given in the above diagram
        Graph g=new Graph(6);
        g.addEdge(5, 2);
        g.addEdge(5, 0);
        g.addEdge(4, 0);
        g.addEdge(4, 1);
        g.addEdge(2, 3);
        g.addEdge(3, 1);
        System.out.println("Following is a Topological Sort");
        g.topologicalSort();

    }}
```
Finding longest path in a DAG[http://www.geeksforgeeks.org/find-longest-path-directed-acyclic-graph/]:

Following is complete algorithm for finding longest distances.
1) Initialize dist[] = {NINF, NINF, ….} and dist[s] = 0 where s is the source vertex. Here NINF means negative infinite.
2) Create a toplogical order of all vertices.
3) Do following for every vertex u in topological order.
………..Do following for every adjacent vertex v of u
………………if (dist[v] < dist[u] + weight(u, v))
………………………dist[v] = dist[u] + weight(u, v)

[From: http://www.mathcs.emory.edu/~cheung/Courses/171/Syllabus/11-Graph/Docs/longest-path-in-dag.pdf]

64. Edit distance: Given two strings str1 and str2 and below operations that can performed on str1. Find minimum number of edits (operations) required to convert ‘str1′ into ‘str2′.
Insert
Remove
Replace
All of the above operations are of equal cost.
Ans:
```cpp
int editDistDP(string str1, string str2, int m, int n)
{
    // Create a table to store results of subproblems
    int dp[m+1][n+1];

    // Fill d[][] in bottom up manner
    for (int i=0; i<=m; i++)
    {
        for (int j=0; j<=n; j++)
        {
            // If first string is empty, only option is to
            // isnert all characters of second string
            if (i==0)
                dp[i][j] = j;  // Min. operations = j

            // If second string is empty, only option is to
            // remove all characters of second string
            else if (j==0)
                dp[i][j] = i; // Min. operations = i

            // If last characters are same, ignore last char
            // and recur for remaining string
            else if (str1[i-1] == str2[j-1])
                dp[i][j] = dp[i-1][j-1];

            // If last character are different, consider all
            // possibilities and find minimum
            else
                dp[i][j] = 1 + min(dp[i][j-1],  // Insert
                                   dp[i-1][j],  // Remove
                                   dp[i-1][j-1]); // Replace
        }
    }

    return dp[m][n];
}
```
Python version:
```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m = len(word1)
        n = len(word2)
        dp = [[0 for j in range(n+1)] for i in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0: # word1 is empty, insert chars into word1
                    dp[i][j] = j
                elif j == 0: # remove chars from word1
                    dp[i][j] = i
                elif word1[i-1] == word2[j-1]:  # no change
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # insert 1char into word1 or remove 1 char from word1 or replace char in word1
                    dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
        return dp[m][n]
```

Follow up easy:
Given two strings S and T, determine if they are both one edit distance apart.
Ans:

/*
* There're 3 possibilities to satisfy one edit distance apart:
*
* 1) Replace 1 char:
       s: a B c
       t: a D c
* 2) Delete 1 char from s:
       s: a D  b c
       t: a    b c
* 3) Delete 1 char from t
       s: a   b c
       t: a D b c
*/
```
public boolean isOneEditDistance(String s, String t) {
    for (int i = 0; i < Math.min(s.length(), t.length()); i++) {
         if (s.charAt(i) != t.charAt(i)) {
              if (s.length() == t.length()) // s has the same length as t, so the only possibility is replacing one char in s and t
                   return s.substring(i + 1).equals(t.substring(i + 1));
               else if (s.length() < t.length()) // t is longer than s, so the only possibility is deleting one char from t
                    return s.substring(i).equals(t.substring(i + 1));
               else // s is longer than t, so the only possibility is deleting one char from s
                    return t.substring(i).equals(s.substring(i + 1));
         }
    }
    //All previous chars are the same, the only possibility is deleting the end char in the longer one of s and t
    return Math.abs(s.length() - t.length()) == 1;
}
```
65.  Huffman encoding [Greedy algorithm]



66

- How is the load balancer works and how could we define the load balancer ?

- Why we need both the IP address and MAC address ? What's the MAC address used for ?

    - MAC addresses are used to send Ethernet frames between two stations in the same local area network. Each station has a unique MAC address that is used to identify who is the sender (source address) and who is the receiver (destination address). But Ethernet frames can't travel between networks. One computer in a local network never sees the MAC of a computer which is on another network.

    - IP addresses are used to send IP packets to another station over the Internet, which is a collection of networks (hence the name "inter networks", from where Internet is derived). Contrary to MAC addresses, IP frames aren't limited to the local network. While travelling around the world, IP packets pass through many smaller networks, many of them using Ethernet (like inside your home or office LAN). When it is the case, the network stack puts the IP packet inside an Ethernet frame, using the MAC address to send to the next stop (what we call 'next hop'). The gateway strips the Ethernet header, rocering the original IP packet, and forwards it over the next network, until it reaches the destination.

- How could we define the load balancer?

    - input: given the machine "A, B, C" and weights "9, 2, 3"
    - let's say the weights mean how much capacity the machine could have
    - write a function to check how to load balance the input request ?
- The simple way we could do is use the random generater generate the value between 0 to 1, and then times the weight of the machine

http://docs.oracle.com/cd/E23943_01/web.1111/e13709/load_balancing.htm#CLUST183
  
-

67. Coin change problem
```python
Set Min[i] equal to Infinity for all of i
Min[0]=0

For i = 1 to S
     For j = 0 to N - 1
            If (Vj<=i AND Min[i-Vj]+1<Min[i])
                    Then Min[i]=Min[i-Vj]+1

Output Min[S]
```
Variant: Given an infinite supply of ‘m’ coin denominations S[m] = {S1, S2... Sm}, calculate all the different combinations which can be used to get change for some quantity ‘N’
So, if N = 4 and S = {1,2,3}, then different ways possible are
{1,1,1,1}, {2,1,1}, {3,1}, {2,2}
Code:
```python
[This will not work]
Set Min[i] equal to 0 for all of i
Min[0]=1
For i = 1 to S //10
     For j = 1 to N
            If (Vj<=i)
                    Then Min[i]=Min[i] + Min[i-Vj]
```
Ans:
```python
    dp[0] = 1
         for coin from c1, c2, .., cm:
              for amount from coin to N:
                    dp[amount] += dp[amount-coin]
```

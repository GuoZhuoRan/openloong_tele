#{1, 5, 7, -1, 5},target = 6

def twoSum(arr,target)-> int:
    n=len(arr)
    cnt=0
    
    for i in range(n):
        for j in range(i+1,n):
            if arr[i]+arr[j]==target:
                cnt+=1

    return cnt

arr = [1, 5, 7, -1, 5]
target = 6

print(twoSum(arr,target))

#Hashmap solution
def count_pairs(arr,target):
    
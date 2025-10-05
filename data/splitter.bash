# make dirs
mkdir -p images/train images/val images/test
mkdir -p labels/train labels/val labels/test

# shuffle list of files
shuf -e images/*.jpg > all.txt

# split counts (80/10/10)
n=$(wc -l < all.txt)
ntrain=$((n*80/100))
nval=$((n*10/100))
ntest=$((n-ntrain-nval))

head -n $ntrain all.txt > train.txt
tail -n +$((ntrain+1)) all.txt | head -n $nval > val.txt
tail -n $ntest all.txt > test.txt

# move files and their labels
while read f; do
  base=$(basename "$f" .jpg)
  mv "$f" images/train/
  mv labels/"$base".txt labels/train/
done < train.txt

while read f; do
  base=$(basename "$f" .jpg)
  mv "$f" images/val/
  mv labels/"$base".txt labels/val/
done < val.txt

while read f; do
  base=$(basename "$f" .jpg)
  mv "$f" images/test/
  mv labels/"$base".txt labels/test/
done < test.txt

# cleanup
rm all.txt train.txt val.txt test.txt


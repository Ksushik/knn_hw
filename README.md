# Homework


## Інсталяція залежностей

```bash
pip install -r requirements.txt
```


## Завдання

1. Допишіть в файлі `kfold.py` функції `kfold_cross_validation` та `evaluate_accuracy` для того щоб порахувати точність роботи KNN алгоритму.

2. Порахуйте для різних `k` в `KNN` точність на **тестовому** датасеті і запишіть в `README.md`, `k` беріть з таблички нижче

 k | Accuracy
---|----------
 3 | 0.82
 4 | 0.8325
 5 | 0.8275
 6 | 0.83
 7 | 0.8075
 9 | 0.8025
10 | 0.81
15 | 0.7925
20 | 0.7775
21 | 0.7775
40 | 0.7325
41 | 0.7325

Які можна зробити висновки про вибір `k`?

Optimal value `k` = 4 

3. Знайшовши найкращий `k` змініть `num_folds` (в `main()`) та подивіться чи в середньому точність на валідаційних датасетах схожа з точністю на тестовому датасеті.

Overall, the model performs reasonably well, however, the drop in accuracy 
from the cross-validation to the test data suggests that the model may be slightly 
overfitting to the training data, as it doesn't generalize quite as well to unseen data.


import numpy as np
import matplotlib.pyplot as plt

class_names = [
    'CatEye', 'DogEye', 'CatFace', 'DogFace',
    'CatMouth', 'DogMouth', 'CatNose', 'DogNose'
]

cm = np.array([
    [2740, 19,   4,    0,   5,   0,   9,   0],
    [19,   2275, 0,    1,   5,   2,   4,  10],
    [6,    1,    1377, 0,   4,   0,   0,   0],
    [0,    3,    0,    1152, 0,   2,   0,   1],
    [8,    2,    1,    1,   817, 3, 556,   0],
    [1,    9,    0,    7,     0, 919, 2, 220],
    [6,    3,    0,    1,   321, 5, 1049,   4],
    [1,   17,    0,    1,     1, 219, 4, 915]
])

fig, ax = plt.subplots(figsize=(12, 9))

# Display raw counts, not normalized values
im = ax.imshow(
    cm,
    interpolation='nearest',
    cmap='Blues',
    vmin=0,
    vmax=cm.max()
)

fig.colorbar(im, ax=ax, label='Number of Predictions')

ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel='Predicted Class',
    ylabel='Actual Class',
    title='Confusion Matrix (Counts)'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Display the actual count in every cell
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        ax.text(
            j, i, f'{value}',
            ha='center',
            va='center',
            color='white' if value > threshold else 'black'
        )

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

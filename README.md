# Capstone design(1)

---
# API usage

## Shape related module

### 개요

Shape related module은 기하학적 모양을 정의하고, 이러한 모양을 패턴으로 변환하며, 그 패턴을 그리드에 시각화하는 기능을 제공합니다. 이 모듈은 기하학적 변환의 시각적 표현을 만들고 이러한 모양의 공간적 배열을 이해하는 데 유용합니다.

### Public API

#### `Shape`

특정 속성과 영역 함수를 가진 기하학적 모양을 정의합니다.

**초기화 인자:**

- **`w`** (float): 모양의 너비.
  - **예시:** `w=10`
- **`h`** (float): 모양의 높이.
  - **예시:** `h=5`
- **`area_functions`** (호출 가능한 함수 목록): 모양의 영역을 정의하는 함수 목록.
  - **예시:**
    ```python
    area_functions=[
        lambda x, y: x**2 / 9 + y**2 / 4 <= 1,  # 타원
        lambda x, y: abs(x / 2) + abs(y) <= 2,  # 다이아몬드
    ]
    ```

**Public 속성:**

- **`w`** (float): 모양의 너비.
- **`h`** (float): 모양의 높이.
- **`area_functions`** (호출 가능한 함수 목록): 모양의 영역을 정의하는 함수들.

**사용 예시:**

```python
# 타원과 다이아몬드를 사용하는 복합 모양 정의
shape = Shape(
    w=10,
    h=5,
    area_functions=[
        lambda x, y: x**2 / 9 + y**2 / 4 <= 1,  # 타원
        lambda x, y: abs(x / 2) + abs(y) <= 2,  # 다이아몬드
    ]
)
```

#### `PatternUnit`

`Shape`를 사용하여 패턴의 단위를 나타냅니다.

**초기화 인자:**

- **`shape`** (Shape): 패턴 단위를 정의하는 Shape 객체.
  - **예시:** `shape=shape`
- **`k`** (float): 그리드를 위한 스케일링 인자.
  - **예시:** `k=0.5`

**Public 속성:**

- **`shape`** (Shape): 패턴 단위를 정의하는 Shape 객체.
- **`k`** (float): 그리드를 위한 스케일링 인자.
- **`shape_matrix`** (numpy.ndarray): 그리드 위의 모양을 나타내는 행렬.
- **`grid`** (Grid): 패턴을 포함하는 그리드 객체.

**사용 예시:**

```python
# 특정 스케일로 모양에서 패턴 단위 생성
pattern_unit = PatternUnit(shape=shape, k=0.5)

# 그리드에 모양 시각화
visualize_points(pattern_unit.shape_matrix, k=pattern_unit.grid.k)
```

#### `PatternTransformation`

변환과 회전 같은 기하학적 변환을 처리합니다.

**초기화 인자:**

- **`name`** (str): 변환의 이름.
  - **예시:** `name='circular'`
- **`dx`** (float): x 방향의 변위.
  - **예시:** `dx=2.5`
- **`dy`** (float): y 방향의 변위 (선택적).
  - **예시:** `dy=2.0`
- **`di`** (float): 원점으로부터의 거리.
  - **예시:** `di=25`
- **`phi`** (float): 라디안 단위의 회전 각도.
  - **예시:** `phi=np.pi / 8`

**Public 속성:**

- **`name`** (str): 변환의 이름.
- **`dx`** (float): x 방향의 변위.
- **`dy`** (float): y 방향의 변위.
- **`di`** (float): 원점으로부터의 거리.
- **`phi`** (float): 회전 각도.

**사용 예시:**

```python
# 원형 변환 정의
circular_transformation = PatternTransformation(
    name='circular',
    dx=2.5,
    di=25,
    phi=np.pi / 8
)
```

#### `PatternTransformationMatrix`

`PatternUnit`에 변환을 적용하여 패턴을 생성합니다.

**초기화 인자:**

- **`pattern_unit`** (PatternUnit): 변환할 패턴 단위.
  - **예시:** `pattern_unit=pattern_unit`
- **`pattern_transformation`** (PatternTransformation): 적용할 변환.
  - **예시:** `pattern_transformation=circular_transformation`
- **`pattern_bound`** (튜플의 튜플): 패턴의 경계.
  - **예시:** `pattern_bound=((-100, 100), (-100, 100))`

**Public 속성:**

- **`pattern_unit`** (PatternUnit): 변환할 패턴 단위.
- **`pattern_transformation`** (PatternTransformation): 적용할 변환.
- **`pattern_bound`** (튜플의 튜플): 패턴의 경계.
- **`transformation_matrix`** (numpy.ndarray): 변환된 패턴을 나타내는 행렬.

**사용 예시:**

```python
# 패턴 단위에 원형 변환 적용
circular_transformation_vector = PatternTransformationMatrix(
    pattern_unit=pattern_unit,
    pattern_transformation=circular_transformation,
    pattern_bound=((-100, 100), (-100, 100))
)
```

#### `Pattern`

변환 행렬을 사용하여 완전한 패턴을 구성합니다.

**초기화 인자:**

- **`pattern_transformation_matrix`** (PatternTransformationMatrix): 사용할 변환 행렬.
  - **예시:** `pattern_transformation_matrix=circular_transformation_vector`

**Public 속성:**

- **`pattern_transformation_matrix`** (PatternTransformationMatrix): 패턴을 만드는 데 사용된 변환 행렬.
- **`pattern_matrix`** (numpy.ndarray): 완전한 패턴을 나타내는 행렬.
- **`pattern_unit`** (PatternUnit): 패턴을 정의하는 데 사용된 패턴 단위.

**사용 예시:**

```python
# 변환 행렬로부터 패턴 생성
circular_pattern = Pattern(
    pattern_transformation_matrix=circular_transformation_vector
)

# 변환된 패턴 시각화
visualize_points(circular_pattern.pattern_matrix, k=circular_pattern.pattern_unit.grid.k)
```

### 추가 참고 사항:

- `Shape` 클래스는 수학적 영역 함수를 사용하여 맞춤형 기하학적 모양을 정의할 수 있게 해줍니다. 이러한 모양은 `PatternTransformation`을 사용하여 변환할 수 있으며 `visualize_points`를 사용하여 시각화할 수 있습니다.
- 이들 모듈은 함께 작동하도록 설계되었습니다. 예를 들어, `visualize_points` 함수는 패턴에 대한 시각적 피드백을 제공하기 위해 Shape와 유전 알고리즘 모듈 모두에서 사용됩니다.

---

## 유전 알고리즘 suite

### 개요

유전 알고리즘 (GA) suite는 최적화 문제를 설정하고 실행하기 위한 프레임워크를 제공합니다. 이 모듈은 개체군의 초기화, 평가, 선택, 교차 및 돌연변이를 처리하여 최적의 솔루션으로 진화합니다.

### Public API

#### `GAPipeline`

유전 알고리즘 실행을 관리하는 주요 클래스.

**초기화 인자:**

- **`suite_name`** (str): GA suite의 이름.
  - **예시:** `suite_name='suite_1'`
- **`suite_max_count`** (int): suite의 최대 카운트.
  - **예시:** `suite_max_count=50`
- **`suite_min_population`** (int): 최소 개체군 수.
  - **예시:** `suite_min_population=10`
- **`population_initializer`** (PopularizationInitializer): 유전 알고리즘의 개체군 초기화자.
  - **예시:** `population_initializer=my_initializer`
- **`fitness_calculator`** (FitnessCalculator): 유전 알고리즘의 피트니스 계산기.
  - **예시:** `fitness_calculator=my_fitness_calculator`
- **`selector_behavior`** (SelectionBehavior): 유전 알고리즘의 선택 행동.
  - **예시:** `selector_behavior=my_selector`
- **`crossover_behavior`** (CrossoverBehavior): 유전 알고리즘의 교차 행동.
  - **예시:** `crossover_behavior=my_crossover`
- **`mutation_probability`** (float): 돌연변이 확률, `[0, 1]` 범위 내.
  - **예시:** `mutation_probability=0.001`
- **`immediate_exit_condition`** (callable): 알고리즘이 즉시 종료해야 하는지 여부를 결정하는 함수.
  - **예시:** `immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000`

**Public 속성:**

- **`suite_name`** (str): GA suite의 이름.
- **`suite_max_count`** (int): suite의 최대 카운트.
- **`suite_min_population`** (int): 최소 개체군 수.
- **`population_initializer`** (PopularizationInitializer): 개체군 초기화자.
- **`fitness_calculator`** (FitnessCalculator): 피트니스 계산기.
- **`selector_behavior`** (SelectionBehavior): 선택 행동.
- **`crossover_behavior`** (CrossoverBehavior): 교차 행동.
- **`mutation_probability`** (float): 돌연변이 확률.
- **`immediate_exit_condition`** (callable): 알고리즘이 즉시 종료해야 하는지 여부를 결정하는 함수.
- **`evolution_storage`** (EvolutionStorage): 진화 데이터를 위한 저장소.
- **`population_storage`** (PopulationStorage): 개체군 데이터를 위한 저장소.
- **`population`** (list): 현재 개체군 목록.
- **`generation`** (int): 현재 세대 번호.
- **`should_stop`** (bool): 알고리즘이 멈춰야 하는지 여부를 나타내는 플래그.
- **`mutation_count`** (int): 수행된 돌연변이 횟수.

**사용 예시:**

```python
from src.ga.genetic_algorithm import GAPipeline, VentFitnessCalculator
from src.ga.p1_initialize.init_popularization import PopularizationInitializer
from src.ga.p3_select.selector_behavior import TournamentSelectionFilter
from src.ga.p4_crossover.crossover_behavior import OnePointCrossover

# 피트니스 기준 및 가중치 정의
criteria_weight_list = (1.2, 0.8, 1)  # 드래그, 최대 온도, 평균 온도 가중치
drag_criterion = ("lower", 0.2, 0.5)  # 드래그는 낮을수록 좋음, 범위 0.2에서 0.5
max_temp_criterion = ("lower", 300, 500)  # 최대 온도는 낮을수록 좋음, 범위 300에서 500
avg_temp_criterion = ("higher", 250, 400)  # 평균 온도는 높을수록 좋음, 범위 250에서 400

# GA 파이프라인 설정
suite1 = GAPipeline(
    suite_name='suite_1',  # GA suite 이름
    suite_max_count=50,  # 실행할 최대 세대 수
    suite_min_population=10,  # 최소 개체군 수
    population_initializer=PopularizationInitializer(
        grid_scale=10**20,  # 그리드 스케일
        grid_resolution=2.0,  # 그리드 해상도
        pattern_bound=((-30, 30), (-30, 30)),  # 패턴 경계
        population_size=100  # 초기 개체군 크기
    ),
    fitness_calculator=VentFitnessCalculator(
        criteria_weight_list,  # 피트니스 기준 가중치
        drag_criterion,  # 드래그 기준
        max_temp_criterion,  # 최대 온도 기준
        avg_temp_criterion  # 평균 온도 기준
    ),
    selector_behavior=TournamentSelectionFilter(tournament_size=5),  # 선택 행동
    crossover_behavior=OnePointCrossover(),  # 교차 행동
    mutation_probability=0.001,  # 돌연변이 확률
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000  # 종료 조건
)

# 유전 알고리즘 실행
suite1.run()

# 피트니스 결과 플롯팅
suite1.evolution_storage.plot_fitness(
    storage="fitness", title="fitness for suite1", xlabel="generation", ylabel="fitness"
)
suite1.evolution_storage.plot_fitness(
    storage="biased_fitness", title="biased fitness for suite1", xlabel="generation", ylabel="biased fitness"
)

# 특정 패턴 단위에 대한 최종 패턴 시각화
for i in range(1):  # 예시로 하나의 패턴 시각화
    visualize_points(
        suite1.population[i].pattern.pattern_matrix,  # 패턴 행렬
        suite1.population[i].pattern.pattern_unit.grid.k  # 그리드 스케일
    )

# 최종 개체군의 세부 정보 출력
for pops in suite1.population:
    print(pops.label)
```

### 추가 참고 사항:

- `GAPipeline` 클래스는 전체 유전 알고리즘 프로세스를 조정합니다. 각 개체군의 피트니스는 `VentFitnessCalculator`를 사용하여 계산되며, 다양한 선택 및 교차 행동을 적용하여 진화 과정을 안내할 수 있습니다. 결과 패턴을 시각화하여 최적화 결과를 이해할 수 있습니다.
- 초기화 매개변수 및 구성을 문제에 맞게 적절히 설정해야 합니다.
- `visualize_points` 함수는 패턴에 대한 시각적 피드백을 제공하기 위해 Shape와 유전 알고리즘 모듈 모두에서 사용됩니다.

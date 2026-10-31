# 전체 June 보정 실행 준비도 — 2026-09-28

**판정:** 전체 June 적격 keys·metadata와 기존 DEV G0 확률은 재사용할 수 있다. 전체 June 104,970행의 모델/frequency 확률 및 calibration label archive는 아직 없으며, 새 추론·보정·비교 비용은 미측정이다. 이번 조사는 설계·사전 등록을 위한 읽기 전용 inventory다. 확률/정답 배열의 값, 새 결과 필드, 모델 checkpoint를 읽거나 역직렬화하지 않았고 fit·추론·검사는 실행하지 않았다. NPZ는 ZIP 내 NPY 헤더의 shape/dtype/배열 이름만 확인했고 파일 SHA256은 opaque bytes로 재확인했다.

조사 시 root HEAD는 `d5466e01bf3ad73262d6c4f4aacc1b43de6c7811`이다. 동결 G0 평가 코드 C는 `5e82bebc239ced378911919de48c106034ee24b0`, 등록 D는 `3f10f61bea591871f69aad6a869d89d0bbe598ed`다. June 적격 준비 C/D는 `53ae9fd0b56650cf076a83919dc1d92cfe2f2da7` / `5025bd79577180bf90850642234bb1ddd8e5fc42`다. 협업은 Astra → Opus → Sol이며 Fable은 이번 범위에서 제외한다.

## 새 설계의 재사용 경계

총괄·Astra가 정한 설계 방향은 **B0=archived G0, B1=전체 June global model–frequency blend, B2=전체 June TRAIN-volume별 blend를 새 B1 weight 쪽으로 1,000구 shrink**다. 새 temperature/class-bias 후보는 없다. B1−B0, B2−B1, B2−B0 세 주 비교를 Holm으로 묶고 R78을 보존하며, 신규 추론을 포함한 전체 비용 한도는 3,600초다. `configs/ML-JUNE-CALIBRATION-v1.json`은 독립 검토한 과학 설정을 담지만 `execution.enabled=false`, 실행 코드 C/등록 D/실행 plan은 null이다. stage caps/profile gate는 과학 계약에 고정하되 실제 비용은 미측정이다. 아직 실행 등록/측정 결과가 아니다.

기존 I1 class-bias와 I2 volume blend는 `EXP-P11-002`의 고정 프로토콜이다. 기존 파일이나 설정을 바꾸지 않고 새 additive adapter를 작성해야 한다. 기존 I2는 **volume temperature가 아니라 model–frequency mixture weight 보정**이다. 기존 May delivery temperatures와 G0 모델·feature·normalizer·encoder·delivery는 그대로 재사용하고 새 June 신경망 fit은 없다.

## 재사용 배열: 값 미열람, shape·identity만 확인

SSD run root는 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924`다. 아래 경로는 이 root 기준이다. 확률은 float64, KEY/label/game/pitcher는 int64이며 class 수는 10이다.

| 봉인 경로 | 사용 가능한 배열 헤더 | 파일 SHA256 |
|---|---|---|
| `EXP-P11-001/analysis/predictions.npz` | `keys [311721,3]`, `y/game_pk/pitcher [311721]`, `raw/calibrated/primary [311721,10]`, `seed_primary [5,311721,10]` | `6890f321e68d0ce83cba203c351a516377c77db6e0491f25802107f237b362d0` |
| `EXP-P11-002/june_inputs.npz` | 기존 **Cpanel만**: `keys [4821,3]`, `y/game_pk/pitcher/groups [4821]`, `frequency/calibrated/primary [4821,10]`, `seed_calibrated/seed_primary [5,4821,10]`, `weights [6]` | `0ea15b692f56940688716d32ad534a372455e6ba9e9670377e379021a06f77e1` |
| `EXP-P10-001/parent_baseline_predictions.npz` | frequency `blend [4821,10]`, `dev [12334,10]`, `mlb_dev [311721,10]`와 각 split의 raw/keys/y/game/pitcher | `bbf4f02a1b6287abff213657a254e4ceaec274d93d59d95041ea5ff6efc9784b` |

기존 Cpanel June member archives는 `blend/blend_raw [4821,10]`, `blend_keys [4821,3]`와 y/game/pitcher/delivery-level을 갖는다. P11 전체 DEV member archives는 `dev/dev_raw [311721,10]`, `dev_keys [311721,3]`와 동일 identity 배열을 갖는다. KEY 배열 일부는 Fortran order이고 분석 archive는 C order이므로 새 row hash는 명시적으로 int64 C-contiguous ordered KEY bytes를 사용해야 한다. 헤더 확인은 행별 KEY 동등성이나 finite/simplex 재검증을 대신하지 않는다.

| seed | 기존 June Cpanel member 경로 | June SHA256 | P11 전체 DEV member SHA256 |
|---|---|---|---|
| 0 | `EXP-P4-001/members/G0-global/seed0/predictions.npz` | `6935d5469177e5ac8883fb47f703890f60779de8b56ade7c1c01c1657270cd10` | `9bc36e1f6271734a1d56e3f8aace13ac0cd3273c9863aef26146805a276d213c` |
| 1 | `EXP-P4-001/members/G0-global/seed1/predictions.npz` | `40ea22c7796acb19ffb736f0fe61f6a4f4d3afada374a4606190fced85088fa6` | `2a7a1c70a1c071aa8be659db13b181d01db1d32320c4d10d8e4fd2923bdd5017` |
| 2 | `EXP-P4-001/members/G0-global/seed2/predictions.npz` | `32d0464554c29c79f7479a1d5ec37b08235d9bdf4496bf00aa8c3e6ff3c68fd9` | `c3986ee970cca5647c3aab5a6955afcc50b306a9ec60ca61aa91cccf498c6e61` |
| 3 | `EXP-P10-001/members/G0-global/seed3/predictions.npz` | `749bab2984330aee7cdf4044bd186ad396d6b528f1bdfa83b7999c6103078fd7` | `ea045bdc2e95db73bc8a4e8c7062fd061927469b49b221237edd7f52740b07e5` |
| 4 | `EXP-P10-001/members/G0-global/seed4/predictions.npz` | `c804aadf4e26d0963b90f77337d5d39d54a3072f7e5862eda82233bf367c26b6` | `877a6c670dcb0e73a4951012699994fad26fc8f64b4688f2ecfba414d4e14149` |

DEV member 경로는 모두 `EXP-P11-001/members/G0-global/seed{seed}/predictions.npz`다. 신규 전체 June inference에서는 기존 Cpanel 4,821행의 각 member raw/calibrated 확률·delivery tier와 ordered KEY를 replay 기준으로 쓸 수 있다. 최종 DEV 적용/채점은 기존 P11 arrays를 재사용할 수 있어 DEV 모델 추론을 다시 할 필요가 없다. 새 과학 계약은 전체 June104,970행을 5회 추론하고 기존 Cpanel을 전체 재현 검사에 쓰는 것으로 고정했다.

## 전체 June 적격 준비의 봉인 metadata

`ML-JUNE-ELIGIBILITY-v1/manifest.json` SHA256은 `f564145deb6b6b58a079766d582573b95154c3515af39fb39b0dba56088bdc88`, `result.json`은 `d00606b8f59d0244cbd8346a09f4b6bf9a2ac45f54be59e6e892891b1095671c`다. 요청 115,816구/397경기에서 적격 **104,970구/394경기/549투수**, 제외 10,846구다. Frozen Cpanel ordered keys 4,821구/110경기를 정확 재현했고, 적격 차집합은 **100,149구/394경기/529투수**다. 이 출력에는 label·description/events·좌표·구종 값이나 확률이 저장돼 있지 않다.

| 파일 | 행 수·ordered KEY SHA256 | 파일 SHA256 |
|---|---|---|
| `eligible_keys.parquet` | 104,970 · `ddb3a0859240821056ea687713b68ab4d2e2d0d8debb8dbd2d64871796e37f82` | `ff9db8e9a6b2f32a9f4279b30ac6a303b0dbb9d9267746f4cb0dc45741d5f037` |
| `eligible_metadata.parquet` | 同上 | `38507b7df491c210ccff46352c47ace4feb385e277c0b88dd3a96b62b5557c76` |
| `complement_eligible_keys.parquet` | 100,149 · `82cddb8a1a90e8a25c48849ba25a47ec94df1fc7c154c1e2bf16a9b53e6d2e12` | `962be8b7e37a6dc89ebb969a8adeab5861c96368d1384d74a567107f57043b43` |
| `complement_eligible_metadata.parquet` | 同上 | `a760cd9e3052f5529ed4ec0854491168a039ef24cf540279e062867b940dba02` |

Frozen P4 TRAIN-volume mapping(q25=170, q75=1514)을 그대로 적용했다. 전체 June 적격 zero/low/middle/high 지원은 각각 **4,788구/127경기, 5,338/184, 38,546/393, 56,298/386**으로 500구 AND 30경기를 모두 충족한다. 차집합은 **4,788/127, 4,703/173, 37,532/393, 53,126/382**다. role×volume `starter|low`와 hand×volume `L|zero`는 이 두 모집단에서 지원 미충족이다. 새 지원 충족은 기존 I2 활성화/성능 판정을 소급 변경하지 않는다. 공유 경기 때문에 행 차집합은 독립 경기 집합이 아니다.

## 코드 재사용과 필요한 새 단계

기존 [June Cpanel loader](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_calibration.py:260)는 frozen member order·KEY/y/game/pitcher·TRAIN lookup을 검증하고 C1 June ensemble/seed weights를 조합한다. 그러나 4,821행만 허용한다. [prepare/fit](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_calibration.py:339)는 Stage2 binding·activation·June inputs를 fit 전 봉인하고 ensemble+seed0–4를 따로 적합한다. [apply](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_calibration.py:516)는 label 없이 기존 DEV 확률에 적용하고 [score](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_calibration.py:596)는 모든 활성 산출물 manifest 후에만 labels를 연다. 이 순서·hash gates는 재사용 가능한 패턴이며 기존 실험의 행 수·ID·후보 의미는 바꾸지 않는다.

[G0 loader](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_whole.py:400)는 `PARTS=('mlb_dev','dev')`와 부모 sample keysets만 지원한다. [load_member](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_whole.py:414)는 checkpoint/seed/network/device·May temperature를 고정하고 [predict](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_g0_whole.py:548)는 member별 arrays·source calibration·dependency hash·replay를 봉인한다. 하위 [sharing loader](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_sharing.py:176)는 cached frame·aux를 열고 5구 history store를 만들며, [regular_frame](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_ml_benchmark.py:110)는 전체 cached regular frame을 준비한다. 새 June 실행에서는 feature/history에 필요한 실제 입력 파일·행/열 창·causal history 정책과 provenance를 별도 등록해야 한다. 적격 parquet만으로 neural 입력을 만들 수 있다고 주장하지 않는다.

아직 없는 실행 계약/산출물은 다음과 같다.

1. 전체 June eligible keys/metadata·정답 source·필수 features/history·frequency 입력을 묶는 새로운 prepare/binding. Label decode 및 feature/cache 접근은 별도 등록 후에만 허용하며 전체 적격 label/KEY hash와 기간·열 노출 기록을 만든다.
2. 고정 5seed frozen G0를 전체 June에 적용하는 profile/비용 gate와 5 member inference. May 보정·400 delivery draws·모델/aux는 유지하고 Cpanel exact replay를 확인한다. 신규 neural fit은 0이다.
3. 모든 June member/frequency/label의 ordered KEY·finite/simplex·seed/source gates 후 ensemble/calibrated/primary inputs를 봉인한다. 새 B1 global blend weights 6개(ensemble+5seed)를 June에서 적합·봉인한다.
4. 새 B2 volume별 blend를 각 predictor의 **새 B1 weight**로 shrink/fallback하는 additive fit/apply. 기존 I2 API가 archived weight를 받는다는 사실만으로 새 B1 provenance를 증명할 수 없으므로 이를 명시적 dependency로 검증해야 한다.
5. 기존 DEV B0 arrays에 새 B1/B2 parameters를 label 없이 적용하고 candidate probability archives를 봉인한다. 새 DEV neural inference는 요구되지 않는다.
6. B1−B0/B2−B1/B2−B0 joint Holm 및 **R78** scorer와 보고. 기존 [robust_bounds](/Users/song/Projects/pitcheezy/experiments/pitchmdp/pitchmdp/matrix_g0_whole_metrics.py:97)는 family size 24 또는 52만 허용하며 [evaluate_candidates](/Users/song/Projects/pitcheezy/experiments/pitchmdp/pitchmdp/matrix_g0_whole_metrics.py:174)는 I1/I2·R52에 고정돼 있다. 새 scorer를 추가하고 기존 24/52 명세는 유지해야 한다.
7. 위 순서를 고정한 단일 supervised queue·family 3,600초 누적 비용·실패 보존·독립 감사. 기존 eligibility 준비 실측1.219838초(cap600)는 완료한 선행비용으로 별도 공개하고 새 family에 재합산하지 않는다. 신규 준비 비용은 전부 새 family에 포함한다.

## 기존 설정·비용과 미측정 항목

새 초안은 B1 6회와 B2 24회의 scalar candidate fit, 기존 Cpanel weights의 identity replay 6회로 **총 36회 optimizer 호출**을 선언한다. 36회는 neural fit이나 36개 후보/HPO가 아니다. 새 bounded scalar 설정은 float64, probability floor1e−12, bounds[0,1], xatol1e−5/maxiter500, choices tie order `[0,optimum.x,1]`, optimizer success·finite/feasible 및 objective tolerance1e−9를 요구한다. 기존 [fit_blend](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_sequence_calibration.py:31)는 동일 bounded 최소화와 endpoint 비교를 쓰지만 반환값에 `optimum.success`, status/nfev, finite/feasible 검사 증거를 담지 않는다. 따라서 함수 반환값만으로 새 `require_success=true`를 입증할 수 없다. 새 adapter에서 optimizer 성공/제약 기록을 보존하고 실패 시 차단하며, 기존 함수/부모를 변경하지 않고 Cpanel 6개 동일성 replay로 legacy 수치 경계를 확인해야 한다.

초안의 primary gates는 whole DEV 세 대비 joint Holm α=.05, ΔNLL≤−.003, 95% NLL CI상한<0, adjusted p≤.05, Brier CI상한≤+.001, matched negative seeds≥4/5, missing slot p=1이다. 10,000 complete-game bootstrap draws/seed20260924와 3대비×13그룹×2손실=R78의 Bonferroni 상한(100,000 draws, α=.05, margins NLL.010/Brier.002)을 담는다. Non-Cpanel은 R의 한 고정 그룹이며 role×volume/hand×volume 교차 그룹은 기술 통계다. 기존 metrics의 R24/R52 helper는 이 계약을 그대로 처리하지 못한다.

추가 부모 identity는 다음처럼 초안에 고정돼 있다. 현재 inventory는 이 metadata/hash를 읽었으며 새 품질 배열을 열지 않았다.

| 부모 경로(root 기준) | SHA256 |
|---|---|
| repo `configs/ML-JUNE-ELIGIBILITY-v1.json` | `afe19b81c0ce58bd5a433f2dda88e0155ce4c605d14c3b4335fa72ffe62e6b30` |
| repo `configs/EXP-P11-001.yaml` | `b6565f167993bdb31481ce18aee6d49276f3db00b61f518b5eee45e8f00a319b` |
| SSD `EXP-P11-001/preparation.json` | `aa831981ed84963e7dc27f7037fed166c2e6f83b0ed29cfba32d2a8b31a0d52a` |
| SSD `EXP-P11-001/analysis/manifest.json` | `aaf5db0d0a3d134b4d92393efa640dc67dba8efb4a4af330d21f2ab66d9ebda9` |
| SSD `EXP-P4-001/preparation.json` | `d544bf38f20686404ed88de241007ddfdad3da2c27bf744d58ad64c6bf0b0423` |
| SSD `EXP-P4-001/panel.json` | `bf779e1d616c7921158816f9de9cafdf86e4015154810fc065265e2f28410f5a` |

기존 `EXP-P11-002.yaml`(SHA256 `75d51ecfacfcf41d8d0519d47f3bc1c60f5e8d4949cf3fac25f568668c3acaad`)의 I1은 floor=1e−12, bias∈[−0.5,+0.5], zero-sum, penalty=0.01, SLSQP zero initialization, ftol=1e−10/maxiter=500, objective/feasibility tolerance=1e−9다. 기존 I2는 4 volume groups, shrinkage=1,000구, 30경기 AND 500구, unsupported exact global-weight fallback, pinned `fit_blend`(log_loss)다. I1/I2 두 primary slot, Holm α=.05, ΔNLL≤−.003, NLL CI 상한<0, Brier CI 상한≤+.001, 음수 seed≥4/5, 10,000 bootstrap draws/seed20260924, R52 100,000 draws와 NLL/Brier margins .010/.002, inactive p=1이었다. 이 숫자들은 **기존 프로토콜의 inventory**이고 새 B1/B2의 확정 config를 대신하지 않는다.

| 측정/참조 | 시간(초) | 해석 |
|---|---:|---|
| June 적격 준비 supervisor Popen→wait | **1.219837540993467** | 실제 115,816 요청/104,970 적격 준비, cap600. 사후 manifest 감사 0.0012692499440163374초와 별도 |
| P11 G0 5 DEV inference 명령 | **1,777.6839637910016** | 311,721행×5seed, 400 draws; load/검증 포함 권위 worker wall |
| P11 profile / 준비 / score | 13.481078709010035 / 2.778572167037055 / 27.30232062493451 | 기존 실제 full-worker 비용 |
| P11 전체 평가 family | **1,821.2459352919832** | 기존 7,200초 cap; outer queue1857.4866603328846초와 합산하지 않음 |
| 기존 Cpanel I1 correction prepare/fit/apply/score | 2.8504363340325654 / 1.7575408339034766 / 7.521853124955669 / 22.29355958290398 | 총34.42338987579569초, 기존 활성 I1만 실행; I2 fit 비용은 미측정 |
| 행 수 선형 참고값: 전체June104,970 또는 차집합100,149 × 기존5seed | 598.623402591232 / 571.1301814433581 | 단순 `1777.6839637910016×N/311721`. 신규 profile·load·replay·저장·fit·R78를 포함한 예측이나 gate가 아님 |
| 새 전체 June 추론·frequency·B1/B2 fit/apply·R78 | **미측정** | 새 profile와 cap/누적 잔여 비용 gate 필요; 3,600초 충족은 아직 확인하지 않음 |

기존 비용 출처는 `coordination/20260927-whole-mlb/execution/attempt-20260927T111115Z/status.json`과 `coordination/20260927-june-eligibility/ledger/ends/june-prepare.json`이다. 내부 timer는 overhead 교차 확인용이며 supervisor 비용에 더하지 않는다.

재사용 코드 SHA256: `run_ml_g0_calibration.py`=`67171381d2739c41d4d3ca844d1e20ffc17769132d58b9c8ddb52799ffffa19f`; `matrix_g0_calibration.py`=`95beee1b33f019e604220997ac12b28924d37f2d93e19793eadcbe1dfc20ec73`; `run_ml_g0_whole.py`=`5430bfbd3849b7f5389ae51194d029e4c48821fdf07252bf999b7cb9db9b0e9a`. G0 연구 동결 번들 SHA256은 `c839dbf710c02ca5a1baafcc168e6ca3eaa6d80e269a99f71f08d7d537ec8062`다. 이 목록은 외부 SSD/실행 worktree 의존을 갖는 참조 inventory이고 self-contained backup이 아니다.

이번 준비가 독립 미개봉 확인셋을 만든 것은 아니다. 기존 2025 DEV 노출과 과거 I1 실패/I2 비활성 기록은 유지하며, B1/B2 새 후보의 효과·독립 확인·정책 우위·서비스 승격은 모두 미측정이다. 2026 자료 접근은 없다.

## 독립 검토 이후 고정한 경계

P4/P10/P11의 기존 frequency archive는 동일 SHA `bbf4f02a…`의 복사본이며, 새 설정은 P10 `parent_baseline_predictions.npz`를 canonical 경로로 사용한다. C1 weights의 생성 소스 `score_ml_matrix.py::summarize_cell`, C1 preparation의 NumPy2.5.3/SciPy1.18.1, 실제 parent/member33개 파일을 설정에 직접 연결했다. 전체 June label/frequency 생성은300초 prepare에 포함되며, 새로운 frequency나 temperature fit은 없다. 프로파일 예측치는 공식 full-worker 비용에104970/8192를 곱해 고정 overhead도 확대한다. 이전1,777.684초는 **5개 별도 worker의 합계**이고 단일 프로세스 amortized 측정이 아니다.

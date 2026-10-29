# arm C(B1) 동결 목록 — EXP-P48-001 주 후보 tabm_residual_counts/MIX_FIT

작성 2026-10-27 17:42 KST · 근거 D188 결정 (a)(Song, 2026-10-27 17:4x KST) · 기계용 원본 [arm-C-B1-freeze-list.json](arm-C-B1-freeze-list.json)

## 한 줄 요약
- EXP-P48-001 Stage B에서 사전 등록한 주 후보가 두 분할 모두 N 규칙을 통과(confirmed)했고, Song이 이 후보를 RS-03(2027 전향 확인 틀) arm C 후보로 기록하기로 했다. 이 문서는 그 후보를 바이트 단위로 고정하는 목록이다. **예측기 번들만 동결**이며 arm C의 정책 식별자(계획기·materialization)는 아직 없다(D187 결정 3: A1 통과 때만 2027 arm 구축).

## 번들 식별자

- bundle_sha256(정렬된 `sha256  path` 35줄의 sha256): `6271b70e834215aff96141b8319e84ba041fb980600cc91791248a1b95f5e4ec`
- Stage B results.json sha256: `54e38dc6734eed202e201e15349b5dd88d55e9506f613512dea0f05b3fa34bc0` (results/EXP-P48-001.json), summary sha256 `fa53f595d93f3aba1da4f491f840bc27b237271cb5909e11b76b1598cf5d361a`
- 등록 커밋 `c5f61d476185`(검토 커밋 `011c257fc29f`, worktree research/direct-head-confirm-1026), 설정 sha `2a417d8f6721d0ff…`, 사전등록 sha `4ca5ba3c79a0daf8…`, export manifest `49097109b4f522a5…`, Stage B 채점기 sha `60c2160b4695c235…`, registry sha `f75bbdcdbda0ca9e…`
- 동결 참조: G15 설정 `b4c19988722c25d9…`, G0 bundle `c839dbf710c02ca5…`, EXP-P16-001 결과 `9d372da26b39fca0…`

## 혼합 규칙(MIX_FIT, 6월 blend 행에서만 선택)

- seed별 head 가중 w: [0.65, 0.7, 0.7, 0.75, 0.7] · 앙상블 w: 0.7 (나머지 1−w는 같은 seed의 동결 G0-global 멤버)

## 동결 파일(tabm_residual_counts, seed 0~4; 맥 사본과 T7 사본 sha256 일치)

| path | bytes | sha256 |
|---|---:|---|
| tabm_residual_counts/0/calibration.json | 5,822 | `4c88f5699c8b5aea0d4d11d1d9c06fd2e19a0660ee2dfc7f91d6a5b39e409eb2` |
| tabm_residual_counts/0/fit.json | 7,580 | `938aa230c564b604fde483dab3e427a28afdc65dbb03b77ef5d88c78e021c683` |
| tabm_residual_counts/0/identity.json | 5,046 | `478e7e0e0f11faf3a04a47645f0a05e9f079c3f995f2d1bd87cc4b399986c9f4` |
| tabm_residual_counts/0/mlb_predictions.npz | 48,322,406 | `368e5c53f7bd61ca38d0f27204768758cd5860a88f2709535ccc6816d2e192d4` |
| tabm_residual_counts/0/model.pt | 1,051,040 | `c32b3f948ff6691955bb89a2b374f50ec91b63069d25a4cb511ffa19c9a7934c` |
| tabm_residual_counts/0/predictions.npz | 2,662,772 | `732526df0a099b5cde6901869edad95489864104ae20fda2ee27fb9fb4248b7f` |
| tabm_residual_counts/0/state.json | 5,834 | `257df29a20dbccce805058097bf904145d776b8bc36db1e2337a8ccf0d498906` |
| tabm_residual_counts/1/calibration.json | 7,423 | `5cfcc1580561f62a708a6e9face14c6461cacb666d4708f2f082a8a98e2ff215` |
| tabm_residual_counts/1/fit.json | 9,301 | `17a1b35967eaef04ab0e2af6926c5721dc43b0c3b293ae9426447d314a26df49` |
| tabm_residual_counts/1/identity.json | 5,046 | `50dd05c5c53e4f6a4987f21de90c66c1643cac9e41ff6d81ef613570346ca56c` |
| tabm_residual_counts/1/mlb_predictions.npz | 48,309,240 | `71520d1e0ef68f6326cb719e4ba9441c9f97f9e5538cf18b8a6fcdc03a5bd28a` |
| tabm_residual_counts/1/model.pt | 1,051,424 | `fe6fbb4c21762155517bb96df938313bad44d5f44b456f9631bd1502fe027ce9` |
| tabm_residual_counts/1/predictions.npz | 2,662,214 | `75b0f5196be1a24e765955b44df1ee7d07662d9497d0a1f5e489a21d2583211e` |
| tabm_residual_counts/1/state.json | 5,834 | `a333fafda72096b92fee8cb3f983357f28ae542532bdd355e85ac12c92b86bec` |
| tabm_residual_counts/2/calibration.json | 5,420 | `8e3f7456dfdb668d96507fa9fa2bcbf8ed3cdeccfe4a2b5868c7f952a13df901` |
| tabm_residual_counts/2/fit.json | 7,148 | `2caaed14280bc10186b478c528ef251712aaf38c4c4e7e856d18da0237d94c89` |
| tabm_residual_counts/2/identity.json | 5,046 | `88e39bd66750afb349c5d4ad30a53b5007a5418d480f27a6a95c4dc1a455c6cf` |
| tabm_residual_counts/2/mlb_predictions.npz | 48,329,540 | `bfd5b897565c5b6ba4740984d249afc9158453e6e29f2fdf3562c0143935682b` |
| tabm_residual_counts/2/model.pt | 1,050,912 | `3ffc695a655f41c1d8bd82ffc43326b94aed248bf0ccef03eef3ba56330d6406` |
| tabm_residual_counts/2/predictions.npz | 2,663,759 | `e29155c70b50f868188de005f1a60db890c4c1dbe47e9e0cf2c11b2c011d4905` |
| tabm_residual_counts/2/state.json | 5,834 | `47ff7f341f9e4ede7167c2e6e24b5369129a5e3f9f782bf1ff99d5b29b0249b4` |
| tabm_residual_counts/3/calibration.json | 8,087 | `1ad0de14d791b07c8c86e5e900feb773acaf0e932fb87f4135426274db2c2dee` |
| tabm_residual_counts/3/fit.json | 10,014 | `05f6fff73fadcdc5d781d61e9405b3f92b7bbed4468d937210089af68b17faa2` |
| tabm_residual_counts/3/identity.json | 5,046 | `ff201571f1dee3efc1fba7e2905da5375dfa5a214457dc2c4c086d937e04702d` |
| tabm_residual_counts/3/mlb_predictions.npz | 48,347,075 | `8ec2b74ac30fd2f0f73cc51dac983b733e4312bb01e118083f77cb2ac7fe025b` |
| tabm_residual_counts/3/model.pt | 1,051,616 | `a12502fb669c1ae5a67aa3c1b617c0dfdaac3250727923da5d23302f15037ecc` |
| tabm_residual_counts/3/predictions.npz | 2,664,060 | `354e2151235317c843e78ca830995f578b15b3fdb1a306d9df8f39c9c08da305` |
| tabm_residual_counts/3/state.json | 5,834 | `5d521aee4ba3e433f9d8eb51a25eefc6c15e8fe47ad3bb484da3065eaf2ca4ab` |
| tabm_residual_counts/4/calibration.json | 5,949 | `e4dfc0acfe982d6d559a71262407639af28d515868a335bffdd80cc59e401e7a` |
| tabm_residual_counts/4/fit.json | 7,716 | `acc76e80cc1e1114c256bbb5c474a0cd4a9ac863280000a4e1f5dc9f9e87f118` |
| tabm_residual_counts/4/identity.json | 5,046 | `944782e8ba6f91818a01177b0d35f8a538efacc1f1aded8b8ca0d57c9002f3ff` |
| tabm_residual_counts/4/mlb_predictions.npz | 48,308,114 | `1e555740c974edc35eb7e5fde55672a2bb752344b967a89740c02b52802b1ce8` |
| tabm_residual_counts/4/model.pt | 1,051,040 | `389acb9616b78ddf5e64224ff0842489956a74a279bc1d1d49295b2c75b0f249` |
| tabm_residual_counts/4/predictions.npz | 2,661,992 | `62d4881a28b2c3c9a7f6680493a634bae9f1b7ccaa49065c604ad70a5b6f891f` |
| tabm_residual_counts/4/state.json | 5,834 | `b868e29ba7cb8f4f522b4d958b3dd7c7c6a97304837b250ddbb9b1e7f11ac673` |

## 위치

- 맥 회수 사본: `/Users/song/gpu-pull/pitcheezy/261027-152429-p48-main/runs/EXP-P48-001`
- T7 사본(보존): `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P48-001/stageA-family` · Stage B: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P48-001/stageB`
- 서버: `~/cc/shared/pitcheezy/direct-confirm-1026/runs/EXP-P48-001`(잡 261027-152429-p48-main (GPU idx2,idx1; commit c5f61d47); 대여 서버라 회수본이 정본)

## 가족 맥락(동결 대상 아님, 식별만)

- control/0/state.json `c7fa242cb75f…`, control/1/state.json `6dce3ef8c106…`, control/2/state.json `29577260e9e2…`, control/3/state.json `fe5e09a60c20…`, control/4/state.json `12be225f7954…`, periodic_residual_counts/0/state.json `1f72d3ed3f3d…`, periodic_residual_counts/1/state.json `05e9b2776094…`, periodic_residual_counts/2/state.json `d7e01b2ccc76…`, periodic_residual_counts/3/state.json `f01d213ce061…`, periodic_residual_counts/4/state.json `a1b4a7242d04…`

## 포함하지 않는 것
- no policy identity (planner, materialization, tau)
- no service artifact
- no 2026 data
- the control and periodic_residual_counts arms are context only

## RS-03 연결
- arm C 포함 조건(RS-03 JSON arms[C].inclusion): Stage B 사전 등록 규칙 통과(confirmed) + Song 결정 줄(D188 결정 (a)) + 동결 목록(이 문서) → 충족. `pins.B1_bundle_sha256`에 위 bundle_sha256을 넣고, `pins.arm_C_identity`는 정책이 구축될 때 채운다.

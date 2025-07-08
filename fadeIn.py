import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def find_first_rise(audio, sample_rate, threshold=0.01, window_size=10):
    """
    波形の最初の立ち上がり部分を検出する。

    Parameters:
    audio (numpy.ndarray): 正規化された音声データ
    sample_rate (int): サンプリングレート
    threshold (float): 立ち上がりと判定する振幅の閾値 (0.0-1.0)
    window_size (int): ノイズ除去のためのウィンドウサイズ

    Returns:
    int: 立ち上がり部分のサンプルインデックス
    float: 立ち上がり部分の時間（秒）
    """
    # ノイズを減らすために移動平均を計算
    smoothed = np.abs(
        np.convolve(audio, np.ones(window_size) / window_size, mode="same")
    )

    # 閾値を超える最初のサンプルを検索
    for i in range(window_size, len(smoothed)):
        if smoothed[i] > threshold and smoothed[i - 1] <= threshold:
            return i, i / sample_rate

    # 見つからない場合は先頭を返す
    return 0, 0.0


def apply_fade_in(audio, sample_rate, start_sample, fade_ms=20):
    """
    特定の位置からフェードイン処理を適用する。

    Parameters:
    audio (numpy.ndarray): 音声データ
    sample_rate (int): サンプリングレート
    start_sample (int): フェードインを開始するサンプル位置
    fade_ms (float): フェードインの長さ（ミリ秒）

    Returns:
    numpy.ndarray: フェードイン処理された音声データ
    """
    # 結果用の配列をコピー
    result = audio.copy()

    # フェードイン範囲のサンプル数を計算
    fade_samples = int(fade_ms * sample_rate / 1000)

    # フェードイン開始位置より前のサンプルを全て0にする
    if start_sample > 0:
        result[:start_sample] = 0

    # フェードイン処理（線形フェード）
    fade_end = min(start_sample + fade_samples, len(audio))
    if fade_end > start_sample:
        fade_curve = np.linspace(0, 1, fade_end - start_sample)
        result[start_sample:fade_end] = audio[start_sample:fade_end] * fade_curve

    return result


def process_audio_with_fade_in(
    input_file, output_file, threshold=0.01, window_size=10, fade_ms=20
):
    """
    音声ファイルを読み込み、立ち上がり位置を検出し、
    その前の値を0にしてからフェードイン処理を適用して保存する。

    Parameters:
    input_file (str): 入力音声ファイルのパス
    output_file (str): 出力音声ファイルのパス
    threshold (float): 立ち上がり検出の閾値
    window_size (int): 立ち上がり検出の移動平均窓サイズ
    fade_ms (float): フェードインの長さ（ミリ秒）

    Returns:
    tuple: (立ち上がり位置のサンプル, 立ち上がり位置の時間)
    """
    # 音声ファイルを読み込む
    sample_rate, audio = wavfile.read(input_file)

    # ステレオの場合の処理
    is_stereo = len(audio.shape) > 1
    if is_stereo:
        channels = audio.shape[1]
        audio_mono = audio[:, 0]  # 左チャンネルを立ち上がり検出に使用
    else:
        audio_mono = audio

    # 正規化（立ち上がり検出用）
    audio_norm = audio_mono / (np.max(np.abs(audio_mono)) + 1e-10)

    # 立ち上がり位置を検出
    rise_sample, rise_time = find_first_rise(
        audio_norm, sample_rate, threshold=threshold, window_size=window_size
    )

    print(f"立ち上がり位置: {rise_time:.3f}秒 ({rise_sample}サンプル)")

    # フェードイン処理の適用
    if is_stereo:
        # ステレオの場合は各チャンネルに適用
        processed_audio = np.zeros_like(audio)
        for ch in range(channels):
            processed_audio[:, ch] = apply_fade_in(
                audio[:, ch], sample_rate, rise_sample, fade_ms=fade_ms
            )
    else:
        # モノラルの場合
        processed_audio = apply_fade_in(
            audio, sample_rate, rise_sample, fade_ms=fade_ms
        )

    # 処理後の音声を保存
    wavfile.write(output_file, sample_rate, processed_audio)

    print(f"処理済み音声ファイルを保存しました: {output_file}")
    return rise_sample, rise_time


def visualize_before_after(input_file, output_file, rise_sample=None, zoom_ms=200):
    """
    処理前と処理後の音声波形を比較表示する。

    Parameters:
    input_file (str): 入力音声ファイルのパス
    output_file (str): 出力音声ファイルのパス
    rise_sample (int): 立ち上がり位置のサンプル（Noneの場合は検出する）
    zoom_ms (float): ズーム表示する範囲（ミリ秒）
    """
    # 音声ファイルを読み込む
    sample_rate_in, audio_in = wavfile.read(input_file)
    sample_rate_out, audio_out = wavfile.read(output_file)

    # ステレオの場合は左チャンネルのみ表示
    if len(audio_in.shape) > 1:
        audio_in = audio_in[:, 0]
    if len(audio_out.shape) > 1:
        audio_out = audio_out[:, 0]

    # 正規化
    audio_in_norm = audio_in / (np.max(np.abs(audio_in)) + 1e-10)
    audio_out_norm = audio_out / (np.max(np.abs(audio_out)) + 1e-10)

    # 立ち上がり位置が指定されていない場合は検出
    if rise_sample is None:
        rise_sample, rise_time = find_first_rise(audio_in_norm, sample_rate_in)
    else:
        rise_time = rise_sample / sample_rate_in

    # 表示範囲の計算
    zoom_samples = int(zoom_ms * sample_rate_in / 1000)
    start_sample = max(0, rise_sample - zoom_samples // 4)
    end_sample = min(len(audio_in), start_sample + zoom_samples)

    # 時間軸
    time = np.linspace(
        0, (end_sample - start_sample) / sample_rate_in, end_sample - start_sample
    )
    time_shift = start_sample / sample_rate_in

    # プロット
    plt.figure(figsize=(14, 8))

    # 処理前
    plt.subplot(2, 1, 1)
    plt.plot(
        time + time_shift,
        audio_in_norm[start_sample:end_sample],
        color="blue",
        label="original",
    )
    plt.axvline(x=rise_time, color="red", linestyle="--", label="apper")
    plt.title("original waveform")
    plt.xlabel("time(sec)")
    plt.ylabel("normalized amplitude")
    plt.grid(True)
    plt.legend()

    # 処理後
    plt.subplot(2, 1, 2)
    plt.plot(
        time + time_shift,
        audio_out_norm[start_sample:end_sample],
        color="green",
        label="processing",
    )
    plt.axvline(x=rise_time, color="red", linestyle="--", label="apper")
    plt.title("post-processing(fade-in)")
    plt.xlabel("time(sec)")
    plt.ylabel("normalized amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.suptitle(f"comparison: apper {rise_time:.3f}s", fontsize=14)
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    # 入出力ファイルのパス
    input_file = "./max/max-original-mono.wav"  # 入力音声ファイル
    output_file = "./max/max-original-mono-processing.wav"  # 出力音声ファイル

    # パラメータ
    threshold = 0.01  # 立ち上がり検出の閾値
    window_size = 10  # 移動平均窓サイズ
    fade_ms = 20  # フェードインの長さ（ミリ秒）

    # 処理の実行
    rise_sample, rise_time = process_audio_with_fade_in(
        input_file,
        output_file,
        threshold=threshold,
        window_size=window_size,
        fade_ms=fade_ms,
    )

    # 結果の可視化
    visualize_before_after(
        input_file, output_file, rise_sample=rise_sample, zoom_ms=200
    )

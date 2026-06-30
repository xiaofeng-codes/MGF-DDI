import subprocess
import sys
import time


def main():
    # for i in [0, 2]:
    #     fold = str(i)
    #     print(f"Calling train.py with argument: {fold}")
    #     subprocess.run([sys.executable, "multiclass/train.py", "--fold", fold])

    max_retries = 5  # 最大重试次数，防止无限死循环

    for dataset in ["zhang"]:
        print(f"Dataset: {dataset}")
        for i in [3]:
            fold = str(i)
            print(f"=== Starting Fold {fold} ===")

            success = False
            attempt = 0

            while not success and attempt < max_retries:
                attempt += 1
                print(f"Running {dataset} Fold {fold} (Attempt {attempt}/{max_retries})...")

                try:
                    # 运行命令
                    result = subprocess.run(
                        [sys.executable, "binary/train.py", "--dataset", dataset, "--fold", fold],
                        check=False  # 这里设为False，手动检查 returncode
                    )

                    # 检查退出代码
                    if result.returncode == 0:
                        print(f"✅ Fold {fold} 完成！")
                        success = True
                    else:
                        print(f"❌ Fold {fold} 报错退出 (Code: {result.returncode})。")
                        if attempt < max_retries:
                            print("⏳ 准备在 10秒 后重试...")
                            time.sleep(10)  # 休息一下，防止如果是过热导致的连续崩溃
                        else:
                            print(f"🚫 Fold {fold} 重试次数耗尽，跳过。")

                except Exception as e:
                    print(f"❌ 调用脚本时发生异常: {e}")
                    time.sleep(5)

    # for dataset in ["miner"]:
    #     print(f"Calling train.py with argument: {dataset}")
    #     for i in [4]:
    #         fold = str(i)
    #         print(f"Calling train.py with argument: {dataset} {fold}")
    #         subprocess.run([sys.executable, "binary/train.py", "--dataset", dataset, "--fold", fold])



if __name__ == "__main__":
    main()


using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    class LinearRegression
    {
        // 假设拟合函数为 h(x) = θ0 + θ1 * x1 + θ2 * x2
        // 通过多元特征值的梯度下降求得各θ值
        // 代价函数 J(θ) = 1/2m ∑{i=1,m} (h(x){i} - y{i})^2
        // 梯度下降 对每个θ同时进行以下处理:
        // θ := θ - α * ∂ J(θ)/∂ θ
        // 即 θ := θ - α * 1/m ∑{i=1,m} ((h(x){i} - y{i}) * x{i})

        //测试样本
        //链家上海桃浦二村永汇新苑 面积、房间数与总价的关系
        struct HousePriceExample
        {
            public int Square;
            public int Rooms;
            public int Price;
            public HousePriceExample(int square, int rooms, int price)
            {
                Square = square;
                Rooms = rooms;
                Price = price;
            }

            public int GetParameter(int index)
            {
                switch (index)
                {
                    case 1:
                        return Square;
                    case 2:
                        return Rooms;
                    default:
                        return 0;
                }
            }
        }

        List<HousePriceExample> examples = new List<HousePriceExample>()
        {
            new HousePriceExample(95, 5, 420),
            new HousePriceExample(78, 3, 370),
            new HousePriceExample(87, 4, 410),
            new HousePriceExample(78, 3, 345),
            new HousePriceExample(99, 5, 475),
            new HousePriceExample(74, 4, 330),
            new HousePriceExample(75, 3, 365),
            new HousePriceExample(85, 4, 385),
            new HousePriceExample(78, 3, 355),
            new HousePriceExample(128, 5, 588),
            new HousePriceExample(80, 4, 390),
            new HousePriceExample(85, 4, 372),
            new HousePriceExample(78, 4, 368),
            new HousePriceExample(79, 3, 380),
            new HousePriceExample(78, 3, 360),
            new HousePriceExample(128, 5, 530),
            new HousePriceExample(73, 4, 355),
            new HousePriceExample(78, 3, 370),
            new HousePriceExample(82, 4, 380),
            new HousePriceExample(51, 2, 250),
            new HousePriceExample(71, 3, 330),
            new HousePriceExample(75, 3, 360),
            new HousePriceExample(84, 4, 380),
            new HousePriceExample(133, 5, 600),
            new HousePriceExample(87, 4, 360),
            new HousePriceExample(78, 4, 370),
            new HousePriceExample(78, 3, 360 ),
        };

        /// <summary>
        /// 迭代次数
        /// </summary>
        public int Iteration = 10000;
        /// <summary>
        /// 步长
        /// </summary>
        public float Step = 0.0001f;

        //首先随机设置初始θ为 10, 5, 2
        public double v1 = 10;
        public double v2 = 5;
        public double v3 = 2;

        public void Run()
        {
            //拟合函数 h(x)
            Func<double, double, double, int, int, double> hypothesis = (_v1, _v2, _v3, squre, rooms) =>
            {
                return _v1 + _v2 * squre + _v3 * rooms;
            };
            //代价函数
            Func<double> cost = () =>
            {
                var doubleExamplesCount = examples.Count * 2;
                double sum = 0;
                foreach (var example in examples)
                {
                    sum += Math.Pow((hypothesis(v1, v2, v3, example.Square, example.Rooms) - example.Price), 2);
                }
                return sum / doubleExamplesCount;
            };
            //梯度下降偏导函数
            Func<double, double, double, int, double> gradientDescent = (_v1, _v2, _v3, index) =>
            {
                double sum = 0;
                foreach (var example in examples)
                {
                    int xi = 1;
                    if (index > 0)
                    {
                        xi = example.GetParameter(index);
                    }
                    sum += (hypothesis(_v1, _v2, _v3, example.Square, example.Rooms) - example.Price) * xi;
                }
                return sum / examples.Count;
            };
            for (int i = 0; i < Iteration; ++i)
            {
                var descent0 = gradientDescent(v1, v2, v3, 0);
                var descent1 = gradientDescent(v1, v2, v3, 1);
                var descent2 = gradientDescent(v1, v2, v3, 2);
                v1 = v1 - Step * descent0;
                v2 = v2 - Step * descent1;
                v3 = v3 - Step * descent2;

                Console.WriteLine(string.Format("本次代价为{0}, 参数分别为{1}、{2}、{3}", cost(), v1, v2, v3));
            }
        }
    }
}

#include <iostream>
#include <csv.hpp> //include the library
#include <string>
#include <memory>
#include <cstdlib>
#include <arrow/api.h>

int main() {
    // Path to your CSV file. Adjust the filename/path as needed.
    std::string filepath = "/Users/anshc/repos/cpp_practice_01/practice01/file-reading/src/stock_data/time-series-AAPL-5min(1).csv";

    // Create a CSVReader using vincentlaucsb-csv-parser.
    // It will automatically parse the CSV rows.
    csv::CSVReader reader(filepath);

    // Create Arrow builders for each column.
    // In this example, we assume:
    //   Column 0: Date (string)
    //   Column 1: Open (double)
    //   Column 2: High (double)
    //   Column 3: Low (double)
    //   Column 4: Close (double)
    //   Column 5: Volume (double)
    arrow::StringBuilder date_builder;
    arrow::DoubleBuilder open_builder;
    arrow::DoubleBuilder high_builder;
    arrow::DoubleBuilder low_builder;
    arrow::DoubleBuilder close_builder;
    arrow::DoubleBuilder volume_builder;

    // Iterate over CSV rows.
    // If your CSV has a header row that you want to skip, make sure to configure your CSVReader accordingly.
    for (csv::CSVRow& row : reader) {
        // Use static typing when reading the fields.
        const std::string date    = row[0].get<>();
        const double open_val     = std::stod(row[1].get<>());
        const double high_val     = std::stod(row[2].get<>());
        const double low_val      = std::stod(row[3].get<>());
        const double close_val    = std::stod(row[4].get<>());
        const double volume_val   = std::stod(row[5].get<>());

        // Append values to the respective builders.
        if (!date_builder.Append(date).ok() ||
            !open_builder.Append(open_val).ok() ||
            !high_builder.Append(high_val).ok() ||
            !low_builder.Append(low_val).ok() ||
            !close_builder.Append(close_val).ok() ||
            !volume_builder.Append(volume_val).ok()) {
            std::cerr << "Error appending to one of the Arrow builders." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Finalize each Arrow array.
    std::shared_ptr<arrow::Array> date_array;
    std::shared_ptr<arrow::Array> open_array;
    std::shared_ptr<arrow::Array> high_array;
    std::shared_ptr<arrow::Array> low_array;
    std::shared_ptr<arrow::Array> close_array;
    std::shared_ptr<arrow::Array> volume_array;

    if (!date_builder.Finish(&date_array).ok() ||
        !open_builder.Finish(&open_array).ok() ||
        !high_builder.Finish(&high_array).ok() ||
        !low_builder.Finish(&low_array).ok() ||
        !close_builder.Finish(&close_array).ok() ||
        !volume_builder.Finish(&volume_array).ok()) {
        std::cerr << "Error finishing one of the Arrow arrays." << std::endl;
        return EXIT_FAILURE;
    }

    // Create an Arrow schema that describes your dataframe.
    std::shared_ptr<arrow::Schema> schema = arrow::schema({
        arrow::field("Date", arrow::utf8()),
        arrow::field("Open", arrow::float64()),
        arrow::field("High", arrow::float64()),
        arrow::field("Low", arrow::float64()),
        arrow::field("Close", arrow::float64()),
        arrow::field("Volume", arrow::float64())
    });

    // Create an Arrow Table from the arrays.
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(
        schema, {date_array, open_array, high_array, low_array, close_array, volume_array});

    // For demonstration, print the schema and the number of rows.
    std::cout << "CSV file has been successfully read into an Arrow Table." << std::endl;
    std::cout << "Schema:\n" << table->schema()->ToString() << std::endl;
    std::cout << "Number of rows: " << table->num_rows() << std::endl;

    // Arrow handles memory management with shared pointers and memory pools.
    // When these objects go out of scope at the end of 'main', memory is safely freed.
    return EXIT_SUCCESS;
}
